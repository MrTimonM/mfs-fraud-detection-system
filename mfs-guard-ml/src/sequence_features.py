"""Past credential/device events followed by the present proposed transaction."""


def sequences(previous, row, historical, age):
    prev = previous if previous is not None and age <= 86400 else {}
    transfer = row['transaction_type'] in ('SEND_MONEY', 'BANK_TRANSFER', 'CASH_OUT')
    cashout = row['transaction_type'] == 'CASH_OUT'
    large = historical['amount_vs_user_mean'] > 2
    result = {
        'pin_reset_then_transfer': bool(prev.get('pin_reset_recently', 0) and transfer),
        'new_device_then_transfer': bool(prev.get('device_is_new', 0) and transfer and large),
        'failed_pin_then_success': bool(prev.get('failed_pin_attempts', 0) >= 2 and transfer),
        'balance_inquiry_then_cashout': bool(prev.get('balance_inquiry_count_5m', 0) and cashout),
        'incoming_then_rapid_outgoing': bool(historical['rapid_fund_turnaround'] and transfer),
        'new_recipient_then_large_transfer': bool(historical['first_time_recipient'] and large and transfer),
        'multiple_otp_then_transfer': bool(prev.get('otp_resend_count', 0) >= 2 and transfer),
        'device_change_then_high_value': bool(prev.get('device_is_new', 0) and large),
        'sim_change_then_transfer': bool(prev.get('sim_changed_recently', 0) and transfer and large),
        'credential_change_then_cashout': bool(prev.get('pin_reset_recently', 0) and cashout),
        'channel_hop_sequence': bool(prev.get('channel_changed_recently', 0) and historical['channel_changed_recently']),
    }
    return {**{k: int(v) for k, v in result.items()}, 'sequence_risk_score': min(100, 12 * sum(result.values()))}
