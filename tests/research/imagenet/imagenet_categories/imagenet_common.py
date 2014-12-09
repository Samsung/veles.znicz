def read_list_from_file(filename):
    with open(filename, 'r') as file:
        items = file.read().splitlines()
    return items


def write_list_to_file(items, filename):
    with open(filename, 'w') as file:
        for item in items:
            file.write(item + "\n")
