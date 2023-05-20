def decimal_to_binary(decimal):
    binary = bin(decimal)[2:]  # Remove the '0b' prefix
    return binary


def binary_to_decimal(binary):
    decimal = int(binary, 2)
    return decimal


# Convert decimal to binary
decimal_number = 1055
binary_number = decimal_to_binary(decimal_number)

decimal_number = binary_to_decimal(binary_number)
print(f"Decimal representation of {binary_number}: {decimal_number}")
