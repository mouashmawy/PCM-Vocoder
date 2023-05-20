def decimal_to_binary(decimal_number, num_bits=32):
    if decimal_number < 0:
        decimal_number = 2 ** num_bits + decimal_number
    
    integer_part = int(abs(decimal_number))
    binary_string = bin(integer_part)[2:].zfill(num_bits)
    
    if "." in str(decimal_number):
        decimal_part = str(decimal_number).split(".")[1]
        decimal_binary = ""
        fractional_part = float("0." + decimal_part)
        while len(decimal_binary) < num_bits - len(binary_string):
            fractional_part *= 2
            bit = int(fractional_part)
            decimal_binary += str(bit)
            fractional_part -= bit
        
        binary_string += decimal_binary
    
    return binary_string


def binary_to_decimal(binary_string, num_bits=32):
    decimal_number = int(binary_string, 2)
    
    if binary_string[0] == "1":
        decimal_number -= 2 ** num_bits
    
    return decimal_number


# # Convert decimal number to binary string
# decimal = -96.5
# binary = decimal_to_binary(decimal, num_bits=32)
# print(binary)  # Output: 11111111111111111111111111010110

# # Convert binary string to decimal number
# decimal = binary_to_decimal(binary, num_bits=32)
# print(decimal)  # Output: -42.75


idx = -55
x = f'{idx:32b}'

print(f'{x}'.format())