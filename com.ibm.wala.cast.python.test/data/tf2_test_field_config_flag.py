# A flag that a constructor sets to a constant and then reads from a parsed configuration, so a
# guard on the flag may be taken although the constant alone would not take it.
import json

import tensorflow as tf


def consume_flagged(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_unguarded(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_method_checked(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def consume_checked(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def check(flag, t):
    if flag:
        consume_checked(t)
        return True
    return False


class Holder:
    def __init__(self, flag=False, config_text=None):
        self.flag = flag
        if config_text is not None:
            config = json.loads(config_text)
            self.flag = config["flag"]

    def run(self, t):
        consume_unguarded(t)
        if self.flag:
            consume_flagged(t)
            return True
        return False

    def check_method(self, flag, t):
        if flag:
            consume_method_checked(t)
            return True
        return False

    def run_method_checked(self, t):
        return self.check_method(self.flag, t)

    def run_checked(self, t):
        return check(self.flag, t)


assert Holder(config_text='{"flag": true}').run_checked(tf.ones([2, 3]))
assert Holder(config_text='{"flag": true}').run_method_checked(tf.ones([2, 3]))
assert Holder(config_text='{"flag": true}').run(tf.ones([2, 3]))


def consume_mode(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def reset_mode():
    global MODE
    MODE = False


def configure_mode(text):
    global MODE
    MODE = json.loads(text)["mode"]


def run_mode(t):
    global MODE
    if MODE:
        consume_mode(t)
        return True
    return False


reset_mode()
configure_mode('{"mode": true}')
assert run_mode(tf.ones([2, 3]))


def consume_phi_checked(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def check_phi(flag, t):
    if flag:
        consume_phi_checked(t)
        return True
    return False


def pick_flag(text):
    flag = json.loads(text)["flag"] if len(text) > 2 else False
    return check_phi(flag, tf.ones([2, 3]))


assert pick_flag('{"flag": true}')


def consume_starred(t):
    assert t.shape == (2, 3) and t.dtype == tf.float32


def check_starred(flag, t):
    if flag:
        consume_starred(t)
        return True
    return False


assert check_starred(*(Holder(config_text='{"flag": true}').flag, tf.ones([2, 3])))


class Pair:
    def __init__(self):
        self.a = False
        self.b = False


def consume_never_seen(t):
    raise AssertionError("never called")


def check_seen(flag, t):
    if flag:
        consume_never_seen(t)
        return True
    return False


def pick_seen(text):
    pair = Pair()
    flag = pair.a if len(text) > 2 else pair.b
    return check_seen(flag, tf.ones([2, 3]))


assert not pick_seen("{}")


def consume_never_counted(t):
    raise AssertionError("never called")


def countdown(flag, t, n):
    if n > 0:
        return countdown(flag, t, n - 1)
    if flag:
        consume_never_counted(t)
        return True
    return False


assert not countdown(False, tf.ones([2, 3]), 2)


def consume_far(t):
    raise AssertionError("never called")


def forward_0(flag, t):
    return forward_1(flag, t)


def forward_1(flag, t):
    return forward_2(flag, t)


def forward_2(flag, t):
    return forward_3(flag, t)


def forward_3(flag, t):
    return forward_4(flag, t)


def forward_4(flag, t):
    return forward_5(flag, t)


def forward_5(flag, t):
    return forward_6(flag, t)


def forward_6(flag, t):
    return forward_7(flag, t)


def forward_7(flag, t):
    return forward_8(flag, t)


def forward_8(flag, t):
    return forward_9(flag, t)


def forward_9(flag, t):
    return forward_10(flag, t)


def forward_10(flag, t):
    return forward_11(flag, t)


def forward_11(flag, t):
    return forward_12(flag, t)


def forward_12(flag, t):
    return forward_13(flag, t)


def forward_13(flag, t):
    return forward_14(flag, t)


def forward_14(flag, t):
    return forward_15(flag, t)


def forward_15(flag, t):
    return forward_16(flag, t)


def forward_16(flag, t):
    return forward_17(flag, t)


def forward_17(flag, t):
    return forward_18(flag, t)


def forward_18(flag, t):
    return forward_19(flag, t)


def forward_19(flag, t):
    return forward_20(flag, t)


def forward_20(flag, t):
    return forward_21(flag, t)


def forward_21(flag, t):
    return forward_22(flag, t)


def forward_22(flag, t):
    return forward_23(flag, t)


def forward_23(flag, t):
    return forward_24(flag, t)


def forward_24(flag, t):
    return forward_25(flag, t)


def forward_25(flag, t):
    return forward_26(flag, t)


def forward_26(flag, t):
    return forward_27(flag, t)


def forward_27(flag, t):
    return forward_28(flag, t)


def forward_28(flag, t):
    return forward_29(flag, t)


def forward_29(flag, t):
    return forward_30(flag, t)


def forward_30(flag, t):
    return forward_31(flag, t)


def forward_31(flag, t):
    return forward_32(flag, t)


def forward_32(flag, t):
    return forward_33(flag, t)


def forward_33(flag, t):
    return forward_34(flag, t)


def forward_34(flag, t):
    return forward_35(flag, t)


def forward_35(flag, t):
    return forward_36(flag, t)


def forward_36(flag, t):
    return forward_37(flag, t)


def forward_37(flag, t):
    return forward_38(flag, t)


def forward_38(flag, t):
    return forward_39(flag, t)


def forward_39(flag, t):
    return forward_40(flag, t)


def forward_40(flag, t):
    return forward_41(flag, t)


def forward_41(flag, t):
    return forward_42(flag, t)


def forward_42(flag, t):
    return forward_43(flag, t)


def forward_43(flag, t):
    return forward_44(flag, t)


def forward_44(flag, t):
    return forward_45(flag, t)


def forward_45(flag, t):
    return forward_46(flag, t)


def forward_46(flag, t):
    return forward_47(flag, t)


def forward_47(flag, t):
    return forward_48(flag, t)


def forward_48(flag, t):
    return forward_49(flag, t)


def forward_49(flag, t):
    return forward_50(flag, t)


def forward_50(flag, t):
    return forward_51(flag, t)


def forward_51(flag, t):
    return forward_52(flag, t)


def forward_52(flag, t):
    return forward_53(flag, t)


def forward_53(flag, t):
    return forward_54(flag, t)


def forward_54(flag, t):
    return forward_55(flag, t)


def forward_55(flag, t):
    return forward_56(flag, t)


def forward_56(flag, t):
    return forward_57(flag, t)


def forward_57(flag, t):
    return forward_58(flag, t)


def forward_58(flag, t):
    return forward_59(flag, t)


def forward_59(flag, t):
    return forward_60(flag, t)


def forward_60(flag, t):
    return forward_61(flag, t)


def forward_61(flag, t):
    return forward_62(flag, t)


def forward_62(flag, t):
    return forward_63(flag, t)


def forward_63(flag, t):
    return forward_64(flag, t)


def forward_64(flag, t):
    return forward_65(flag, t)


def forward_65(flag, t):
    return forward_66(flag, t)


def forward_66(flag, t):
    return forward_67(flag, t)


def forward_67(flag, t):
    return forward_68(flag, t)


def forward_68(flag, t):
    return forward_69(flag, t)


def forward_69(flag, t):
    return forward_70(flag, t)


def forward_70(flag, t):
    if flag:
        consume_far(t)
        return True
    return False


assert not forward_0(False, tf.ones([2, 3]))
