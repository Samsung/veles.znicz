from veles.znicz.tests.research.imagenet_categories.imagenet_categories \
    import ImageNetCategories

name = "basketball"

categories = ImageNetCategories()

index = categories.name_to_index(name)
print("Index for 'basketball' category: " + index)
print("Description for " + name + ": " + categories.name_to_description(name))

print()

print("Name for " + index + ": " + categories.index_to_name(index))
print("Description for " + index + ": " +
      categories.index_to_description(index))

print()


indices_list = ["n01503061", "n02128669", "n03640166"]

indices_and_names_list = categories.indices_list_convertion(indices_list,
                                                            True, False)
indices_and_descriptions_list = categories.indices_list_convertion(
    indices_list, False, True)
indices_and_names_descriptions_list = categories.indices_list_convertion(
    indices_list, True, True)

for elem in indices_and_names_list:
    print(elem)

print("or")

for elem in indices_and_descriptions_list:
    print(elem)

print("or")

for elem in indices_and_names_descriptions_list:
    print(elem)

print()

print("Direct subcategories of 'rock, stone' category:")
humans_subcategories = categories.get_subcategories(categories.name_to_index(
    "rock, stone"))

for index in humans_subcategories:
    print(categories.index_to_name(index))

print()

print("Count of humans-related category with all subcategories (recursive):")
print(len(categories.get_subcategories(categories.name_to_index(
    "person, individual, someone, somebody, mortal, soul"), True, True)))

print()
