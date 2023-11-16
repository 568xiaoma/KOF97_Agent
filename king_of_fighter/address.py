from MAMEToolkit.emulator import Address

def setup_memory_addresses():
    memory = {
        "playing": Address('0x10A83E', 'u8'),

        "1P_input": Address('0x300000', 'u8'), # 单帧按下
        "1P_2Frame": Address('0x10DA44', 'u8'), # 两帧与同时按下
        "healthP1": Address('0x108239', 's8'),
        "1P_x": Address('0x108118', 'u16'),
        "1P_y": Address('0x108120', 'u16'),
        "1P_AttackStatus": Address("0x1081aa", "u32"),
        "1P_Status": Address("0x1081e0", "u8"),
        "1P_StatusExtra": Address("0x1081e1", "u8"),
        "1P_PowerValue": Address("0x1082e3", "u8"),
        "1P_PowerStatus": Address("0x10825f", "u8"),
        
        "2P_input": Address('0x340000', 'u8'), # 单帧按下
        "2P_2Frame": Address('0x10DC44', 'u8'), # 两帧与同时按下
        "healthP2": Address('0x108439', 's8'),
        "2P_x": Address('0x108318', 'u16'),
        "2P_y": Address('0x108320', 'u16'),
        "2P_AttackStatus": Address("0x1083aa", "u32"),
        "2P_Status": Address("0x1083e0", "u8"),
        "2P_StatusExtra": Address("0x1083e1", "u8"),
        "2P_PowerValue": Address("0x1084e3", "u8"),
        "2P_PowerStatus": Address("0x10845f", "u8"),

        # match time
        "time": Address('0x10A83A', 'u8'),
    }
    return memory