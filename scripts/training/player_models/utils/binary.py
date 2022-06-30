import os
import numpy as np

from utils.deal import DealMeta, seats, seat_index, suit_index_lookup

card_index_lookup = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

card_index_lookup_x = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7],
    )
)


def binary_hand(suits):
    x = np.zeros(32, np.float16)
    assert (len(suits) == 4)
    for suit_index in [0, 1, 2, 3]:
        for card in suits[suit_index]:
            card_index = card_index_lookup_x[card]
            x[suit_index * 8 + card_index] += 1
    assert (np.sum(x) == 13)
    return x


def get_cards(play_str):
    cards = []
    i = 0
    while i < len(play_str):
        cards.append(play_str[i:i + 2])
        i += 2
    return cards


def get_tricks(cards):
    return list(map(list, np.array(cards).reshape((13, 4))))


def get_card_index(card):
    suit, value = card[0], card[1]
    return suit_index_lookup[suit] * 8 + card_index_lookup_x[value]


def encode_card(card):
    x = np.zeros(32, np.float16)
    if card == '>>':
        return x
    x[get_card_index(card)] = 1
    return x


def wins_trick_index(trick, trump, lead_index):
    led_suit = trick[0][0]
    card_values = []
    for card in trick:
        suit, value = card[0], 14 - card_index_lookup[card[1]]
        if suit == trump:
            card_values.append(value + 13)
        elif suit == led_suit:
            card_values.append(value)
        else:
            card_values.append(0)
    return (np.argmax(card_values) + lead_index) % 4


def get_play_labels(play_str, trump, player_turn_i):
    tricks = get_tricks(get_cards(play_str))

    trick_ix, leads, last_tricks, cards_in, labels = [], [], [], [], []

    lead_index = 0
    prev_lead_index = 0
    last_trick = ['>>', '>>', '>>', '>>']
    for trick_i, trick in enumerate(tricks):
        last_tricks.append(last_trick)
        leads.append(prev_lead_index)

        current_trick = ['>>', '>>', '>>']

        for i, card in enumerate(trick):
            player_i = (lead_index + i) % 4

            if player_i == player_turn_i:  # the player for whom we generate data is on play
                labels.append(card)
                trick_ix.append(trick_i)
                cards_in.append(current_trick)
                break
            else:
                current_trick.append(card)
                del current_trick[0]

        if lead_index == 0:
            last_trick = trick
        elif lead_index == 1:
            last_trick = trick[3:] + trick[:3]
        elif lead_index == 2:
            last_trick = trick[2:] + trick[:2]
        else:
            last_trick = trick[1:] + trick[:1]
        prev_lead_index, lead_index = lead_index, wins_trick_index(trick, trump, lead_index)

    return trick_ix, leads, last_tricks, cards_in, labels


def play_data_iterator(fin):
    lines = []
    for i, line in enumerate(fin):
        line = line.strip()
        if i % 4 == 0 and i > 0:
            yield (lines[0], lines[1], lines[3])
            lines = []

        lines.append(line)

    yield (lines[0], lines[1], lines[3])


def binary_data(deal_str, outcome_str, play_str, position):
    """
    :param position Position of the player (0 for declarer, 1 for lefty, 2 for dummy, 3 for righty)

    ## input
      0:32  player hand
     32:64  seen hand (dummy, or declarer if we create data for dummy)
     64:96  last trick card 0
     96:128 last trick card 1
    128:160 last trick card 2
    160:192 last trick card 3
    192:224 this trick lho card
    224:256 this trick pard card
    256:288 this trick rho card
    288:292 last trick lead player index one-hot
        292 level
    293:298 strain one hot N, S, H, D, C

    ## output
    0:32 card to play in current trick
    """
    x = np.zeros((1, 11, 298), np.float16)
    y = np.zeros((1, 11, 32), np.float16)

    hands = list(map(lambda hand_str: list(map(list, hand_str.split('.'))), deal_str[2:].split()))
    d_meta = DealMeta.from_str(outcome_str)

    declarer_i = seat_index[d_meta.declarer]
    dummy_i = (declarer_i + 2) % 4
    me_i = (declarer_i + position) % 4

    me_bin = binary_hand(hands[me_i])
    visible_bin = binary_hand(hands[dummy_i])

    # if dummy is actual player, dummy see the declarer hand
    if dummy_i == me_i:
        visible_bin = binary_hand(hands[declarer_i])

    _, on_leads, last_tricks, cards_ins, card_outs = get_play_labels(play_str, d_meta.strain, (position - 1) % 4)

    visible_played_cards = set(['>>'])

    for i, (on_lead, last_trick, cards_in, card_out) in enumerate(zip(on_leads, last_tricks, cards_ins, card_outs)):
        if i > 10:
            break

        label_card_ix = get_card_index(card_out)
        y[0, i, label_card_ix] = 1
        x[0, i, 292] = d_meta.level
        if d_meta.strain == 'N':
            x[0, i, 293] = 1
        else:
            x[0, i, 294 + suit_index_lookup[d_meta.strain]] = 1

        x[0, i, 288 + on_lead] = 1

        last_trick_visible_card = last_trick[3 if dummy_i == me_i else 1]
        if last_trick_visible_card not in visible_played_cards:
            visible_bin[get_card_index(last_trick_visible_card)] -= 1
            visible_played_cards.add(last_trick_visible_card)

        cards_in_i = (position - 1) % 4
        if me_i == declarer_i or me_i == dummy_i:
            cards_in_i = 1

        if cards_in[cards_in_i] not in visible_played_cards:
            visible_bin[get_card_index(cards_in[cards_in_i])] -= 1
            visible_played_cards.add(cards_in[cards_in_i])

        x[0, i, 32:64] = visible_bin
        x[0, i, 0:32] = me_bin

        x[0, i, 64:96] = encode_card(last_trick[0])
        x[0, i, 96:128] = encode_card(last_trick[1])
        x[0, i, 128:160] = encode_card(last_trick[2])
        x[0, i, 160:192] = encode_card(last_trick[3])

        x[0, i, 192:224] = encode_card(cards_in[0])
        x[0, i, 224:256] = encode_card(cards_in[1])
        x[0, i, 256:288] = encode_card(cards_in[2])

        me_bin[label_card_ix] -= 1

    return x, y


def create_binary(data_it, position, n, out_dir, is_train_file):
    """
    :param position Position of the player (0 for declarer, 1 for lefty, 2 for dummy, 3 for righty)
    """

    x = np.zeros((n, 11, 298), np.float16)
    y = np.zeros((n, 11, 32), np.float16)

    for i, (deal_str, outcome_str, play_str) in enumerate(data_it):
        if i % 10000 == 0:
            print(str(i) + '/' + str(n))

        x_i, y_i = binary_data(deal_str, outcome_str, play_str, position)

        x[i, :, :] = x_i
        y[i, :, :] = y_i

    np.save(os.path.join(out_dir, 'X_train.npy' if is_train_file else 'X_val.npy'), x)
    np.save(os.path.join(out_dir, 'Y_train.npy' if is_train_file else 'Y_val.npy'), y)


def execute_binary_script(train_file, valid_file, out_dir, position):
    nb_train_lines = int(sum(1 for _ in open(train_file)) / 4)
    nb_valid_lines = int(sum(1 for _ in open(valid_file)) / 4)

    print("Generating training binary files")
    create_binary(play_data_iterator(open(train_file)), position, nb_train_lines, out_dir, True)

    print("Generating validating binary files")
    create_binary(play_data_iterator(open(valid_file)), position, nb_valid_lines, out_dir, False)
