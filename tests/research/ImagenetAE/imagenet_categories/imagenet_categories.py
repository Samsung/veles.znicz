# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on June 1, 2014

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


import os
from veles.config import root

INDICES_TO_NAMES_FILE = os.path.join(
    root.common.dirs.datasets, "imagenet/2014/indices_to_categories.txt")
INDICES_TO_DESCRIPTIONS_FILE = os.path.join(
    root.common.dirs.datasets, "imagenet/2014/indices_to_descriptions.txt")
INDICES_HIERARCHY_FILE = os.path.join(
    root.common.dirs.datasets, "imagenet/2014/indices_hierarchy.txt")


def write_list_to_file(items, filename):
    with open(filename, 'w') as file:
        for item in items:
            file.write(item + "\n")


class ImageNetCategories(object):

    lists_loaded = False

    def load_indices_to_names(self):
        indices_to_names = {}
        with open(INDICES_TO_NAMES_FILE) as file:
            lines = file.read().splitlines()
            for line in lines:
                pair = line.split("\t")
                indices_to_names[pair[0]] = pair[1]

        return indices_to_names

    def load_indices_to_descriptions(self):
        indices_to_names = {}
        with open(INDICES_TO_DESCRIPTIONS_FILE) as file:
            lines = file.read().splitlines()
            for line in lines:
                pair = line.split("\t")
                indices_to_names[pair[0]] = pair[1]

        return indices_to_names

    def load_indices_hierarchy(self):
        hierarchy = {}
        with open(INDICES_HIERARCHY_FILE) as file:
            lines = file.read().splitlines()
            for line in lines:
                category_and_subcategories = line.split(" ")
                hierarchy[category_and_subcategories[0]] = \
                    category_and_subcategories[1:]

        return hierarchy

    def index_to_name(self, index):

        if index in self.indices_to_names:
            return self.indices_to_names[index]
        else:
            return "UNDEFINED"

    def index_to_description(self, index):
        if index in self.__class__.indices_to_descriptions:
            return self.__class__.indices_to_descriptions[index]
        else:
            return "UNDEFINED"

    def name_to_index(self, name):
        if name in self.__class__.names_to_indices:
            return self.__class__.names_to_indices[name]
        else:
            return "UNDEFINED"

    def name_to_description(self, name):
        if name in self.__class__.names_to_indices:
            index = self.__class__.names_to_indices[name]
            if index in self.__class__.indices_to_descriptions:
                return self.__class__.indices_to_descriptions[index]
            else:
                return "UNDEFINED"
        else:
            return "UNDEFINED"

    def indices_list_convertion(self, indices, include_names,
                                include_descriptions, printList=False):

        items = []
        for index in indices:
            items += [index]

            # if has description
            if (index in self.__class__.indices_to_names):
                if include_names:
                    items[-1] += "\t" + self.index_to_name(index)
                if include_descriptions:
                    items[-1] += "\t - " + self.index_to_description(index)
            if printList:
                print(items[-1])
        return items

    def directories_as_indices_conversion(self, inputDirectory, include_names,
                                          include_descriptions, print_list):
        indices = os.listdir(inputDirectory)
        return self.indices_list_convertion(indices, include_names,
                                            include_descriptions, print_list)

    def get_subcategories(self, index, recursive=False, include_parent=False):
        subcategories = []
        if include_parent:
            subcategories += [index]
        if not recursive:
            subcategories += self.__class__.hierarchy[index]
            return subcategories
        else:
            stack = [index]
            while(len(stack) > 0):
                cur_category = stack.pop()
                new_subcategories = self.__class__.hierarchy[cur_category]
                stack += new_subcategories
                subcategories += new_subcategories

            return subcategories

    def __init__(self):
        if not self.__class__.lists_loaded:
            # load lists for convertions for same all instances
            self.__class__.indices_to_names = self.load_indices_to_names()
            self.__class__.indices_to_descriptions = \
                self.load_indices_to_descriptions()
            self.__class__.names_to_indices = \
                {v: k for k, v in self.__class__.indices_to_names.items()}
            self.__class__.descriptions_to_indices = \
                {v: k for k, v in
                    self.__class__.indices_to_descriptions.items()}
            self.__class__.hierarchy = self.load_indices_hierarchy()
            self.__class__.lists_loaded = True
