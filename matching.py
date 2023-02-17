# import pprint
# from os import listdir
# from os.path import isfile
#
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import dateparser
# import re
# import pickle
#
# import tqdm
# from pymongo import MongoClient
# import json
# import os
# from valentine import valentine_match
# from valentine.algorithms import Coma, Cupid
# from lxml import etree
# import datetime
# import gensim.downloader
# import wordninja
# from nltk import download
# from nltk.corpus import stopwords
# from joblib import Parallel, delayed
# import time
#
#
# def check_value_in_dict(val, dict_to_check, res_list, path=None):
#     for i in dict_to_check:
#         if path is None:
#             aux_path = i
#         else:
#             aux_path = path + "." + i
#         # if isinstance(val, list):
#         #     if dict_to_check[i] == val or dict_to_check[i] == val:
#         #         pass
#         if dict_to_check[i] == val:
#             res_list.append(aux_path)
#         if isinstance(dict_to_check[i], dict):
#             check_value_in_dict(val, dict_to_check[i], res_list, aux_path)
#
#
# def map_two_dict(dict_a, dict_b, res_tuple_list, path=None):
#     for path_iterator_var in dict_a:
#         if isinstance(dict_a[path_iterator_var], dict):
#             if path is None:
#                 path_aux = path_iterator_var
#             else:
#                 path_aux = path + "." + path_iterator_var
#
#             map_two_dict(dict_a[path_iterator_var], dict_b, res_tuple_list, path_aux)
#         else:
#             if isinstance(dict_a[path_iterator_var], list):
#                 contains_dict = False
#                 for index, element in enumerate(dict_a[path_iterator_var]):
#                     if isinstance(element, dict):
#                         contains_dict = True
#                         if path is None:
#                             path_aux = path_iterator_var
#                         else:
#                             path_aux = path + "." + str(index) + "." + path_iterator_var
#                         map_two_dict(element, dict_b, res_tuple_list, path_aux)
#                 if not contains_dict:
#                     for index, element in enumerate(dict_a[path_iterator_var]):
#                         list_aux = []
#                         check_value_in_dict(element, dict_b, list_aux)
#                         if list_aux:
#                             if path is None:
#                                 res_tuple_list.append((path_iterator_var, list_aux))
#                             else:
#                                 res_tuple_list.append((path + "." + str(index) + "." + path_iterator_var, list_aux))
#             list_aux = []
#             check_value_in_dict(dict_a[path_iterator_var], dict_b, list_aux)
#             if list_aux:
#                 if path is None:
#                     res_tuple_list.append((path_iterator_var, list_aux))
#                 else:
#                     res_tuple_list.append((path + "." + path_iterator_var, list_aux))
#
#
# def get_all_keys(doc, main_key=None, separator=".", key_list=[], first_call=True):
#     '''
#     Get all key and sub key of a document, sub key are constructed with a separator defined as a parameter.
#     :param doc:
#     :param main_key:
#     :param separator:
#     :param first_call
#     :return:
#     '''
#     if isinstance(doc, type({})):
#
#         for key in doc.keys():
#             if main_key is None:
#                 key_list.append(key)
#                 get_all_keys(doc[key], main_key=key, separator=separator, key_list=key_list, first_call=False)
#             else:
#                 key_list.append(main_key + separator + key)
#                 get_all_keys(doc[key], main_key=main_key + separator + key, separator=separator, key_list=key_list,
#                              first_call=False)
#     if isinstance(doc, type([])):
#         for obj in doc:
#             if isinstance(obj, type({})):
#                 for key in obj.keys():
#                     if main_key is None:
#                         key_list.append(key)
#                         get_all_keys(obj[key], main_key=key, separator=separator, key_list=key_list, first_call=False)
#                     else:
#                         key_list.append(main_key + separator + key)
#                         get_all_keys(obj[key], main_key=main_key + separator + key, separator=separator,
#                                      key_list=key_list, first_call=False)
#
#     return key_list
#
#
# def remove_id_in_json(f_json):
#     """
#     As mongoDB id or _id is reserved keyword, this function modify any field that is "id" or "_id" to add "doc" before.
#     :param f_json: dict containing a parsed json
#     :return: dict : f_json with a modified "id" or "_id" key if there was
#     """
#     key_list = ["id", "_id"]
#     if index_key_list := [index for index, key_is_present in enumerate([key in f_json for key in key_list]) if
#                           key_is_present]:
#         for index in index_key_list:
#             f_json["doc_" + key_list[index]] = f_json.pop(key_list[index])
#     return f_json
#
#
# def format_date_json(doc):
#     if type(doc) is dict:
#         for key in doc.keys():
#             if (type(doc[key]) is str) and (date := dateparser.parse(doc[key])) is not None:
#                 doc[key] = date
#             format_date_json(doc[key])
#     if type(doc) is list:
#         for object in doc:
#             format_date_json(object)
#
#     return doc
#
#
# def read_preprocess_insert_in_mongodb_json(fp, mongodb_coll=None, fp_is_dict=False):
#     """
#     Read, remove any incompatible "id" key and insert the JSON in a collection in a mongodb database
#     :param fp: str : file_path to a JSON to read
#     :param mongodb_coll: MongoClient.database.collection : A collection in which insert files
#     :return: None
#     """
#     if mongodb_coll is None:
#         mongodb_coll = MongoClient("localhost:27017").no_model_name.interop_metadata
#     try:
#         # mongodb_coll.insert_one(format_date_json(remove_id_in_json(json.load(open(fp)))))
#         # Error seens in data formatting
#         if fp_is_dict:
#             mongodb_coll.insert_one((remove_id_in_json(fp)))
#         else:
#             mongodb_coll.insert_one((remove_id_in_json(json.load(open(fp)))))
#         # print(fp + " has been inserted successfully.")
#     except Exception as exce:
#         print("Insertion has not been successfully done. Logs : " + str(exce))
#
#
# def dont_contains_dict(liste):
#     for elem in liste:
#         # print(elem)
#         if type(elem) is dict:
#             return False
#     return True
#
#
# # Transform a date to standard format
# def format_date(date):
#     # Transform a string date into a standard format by trying each
#     # date format. If you want to add a format, add a try/except in the
#     # last except
#     # date : str : the date to transform
#     # return : m : timedata : format is YYYY-MM-DD HH:MM:SS
#     date_str = date
#     #
#     date_str = date_str.replace("st", "").replace("th", "") \
#         .replace("nd", "").replace("rd", "").replace(" Augu ", " Aug ")
#     m = None
#     sep_list = [".", "/", "-", "_", " ", ":"]
#     for date_sep in sep_list:
#         try:
#             m = datetime.datetime.strptime(date_str, "%d" + date_sep + "%B" + date_sep + "%Y")
#             break
#         except ValueError:
#             try:
#                 m = datetime.datetime.strptime(date_str, "%d" + date_sep + "%b" + date_sep + "%Y")
#                 break
#             except ValueError:
#                 try:
#                     m = datetime.datetime.strptime(date_str, "%Y" + date_sep + "%m" + date_sep + "%d")
#                     break
#                 except ValueError:
#                     try:
#                         m = datetime.datetime.strptime(date_str,
#                                                        "%d" + date_sep + "%m" + date_sep + "%Y")
#                         break
#                     except ValueError:
#                         for hour_sep in sep_list:
#                             try:
#                                 m = datetime.datetime \
#                                     .strptime(date_str,
#                                               "%d" + date_sep + "%m" + date_sep + "%Y %H" + hour_sep + "%M" + hour_sep + "%S")
#                                 break
#                             except ValueError:
#                                 try:
#                                     m = datetime.datetime \
#                                         .strptime(date_str,
#                                                   "%Y" + date_sep + "%m" + date_sep + "%d %H" + hour_sep + "%M" + hour_sep + "%S")
#                                     break
#                                 except ValueError:
#                                     # HERE ADD A FORMAT TO CHECK
#                                     # print("Format not recognised. \nConsider "
#                                     #       "adding a date format "
#                                     #       "in the function \"format_date\".")
#                                     pass
#
#     return m
#
#
# def from_keyset_to_csv(key_set, separator="."):
#     '''
#     :param key_set:
#     :param separator:
#     :return:
#     '''
#     color = {"0": "#F3722C", "1": "#F8961E", "2": "#F9C74F", "3": "#90BE6D", "4": "#43AA8B", "5": "#4D908E",
#              "6": "#577590", "7": "#277DA1", "8": "#bdd5ea", "9": "#e3e2e3", "10": "#ffffff", "11": "#ffffff"}
#     csv_list = [("ROOT_NODE", "", -1, "#F94144")]
#     for key_concat in key_set:
#         key_split = key_concat.split(separator)
#         for index in range(len(key_split)):
#             if index < 9:
#                 if len(key_split) == 1:
#                     csv_list.append((separator.join(key_split[:index + 1]), "ROOT_NODE", index, color[str(index)]))
#                     break
#                 if index == len(key_split) - 1:
#                     break
#                 csv_list.append((
#                     separator.join(key_split[:index + 2]), separator.join(key_split[:index + 1]), index + 1,
#                     color[str(index + 1)]))
#
#             else:
#                 if index == len(key_split) - 1:
#                     break
#                 csv_list.append((
#                     separator.join(key_split[:index + 2]), separator.join(key_split[:index + 1]), index + 1,
#                     "#ffffff"))
#     return (["key", "mother_key", "level", "color"], set(csv_list))
#
#
# def get_leaf(node_list):
#     aux_json = {}
#
#     def merge(d1, d2):
#         for k in d2:
#             if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
#                 merge(d1[k], d2[k])
#             else:
#                 d1[k] = d2[k]
#
#     def create_dict_from_label(token_list):
#         aux = {}
#         token = token_list.pop(0)
#         if token_list:
#             aux[token] = create_dict_from_label(token_list)
#         else:
#             aux[token] = {}
#         return aux
#
#     for i in node_list:
#         merge(aux_json, create_dict_from_label(i.split("->")))
#
#     def paths(tree, cur=""):
#         if not tree:
#             yield cur
#         else:
#             for n, s in tree.items():
#                 if cur == "":
#                     for path in paths(s, n):
#                         yield path
#                 else:
#                     for path in paths(s, cur + "->" + n):
#                         yield path
#
#     return list(paths(aux_json))
#
#
# def XML_to_dict(xml):
#     xml_tag = re.sub("{.*}", "", xml.tag)
#     res_dict = {xml_tag: {}}
#     path = [xml_tag]
#
#     def tree_walk(xml, path):
#         aux_path = path
#         for child in xml.getchildren():
#             child_tag = re.sub("{.*}", "", child.tag)
#             aux = res_dict
#             for i in path:
#                 aux = aux[i]
#             if child_tag in aux:
#                 for attr in child.attrib:
#
#                     if re.sub("{.*}", "", attr) in aux[child_tag]:
#
#                         if isinstance(aux[child_tag][re.sub("{.*}", "", attr)], list):
#                             aux[child_tag][re.sub("{.*}", "", attr)].append(child.attrib[attr])
#                         else:
#                             aux[child_tag][re.sub("{.*}", "", attr)] = [aux[child_tag][re.sub("{.*}", "", attr)]] + [
#                                 child.attrib[attr]]
#                     else:
#                         aux[child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
#                 if isinstance(child.text, str):
#                     if child.text.strip():
#                         if "@value" in aux[child_tag]:
#                             if isinstance(aux[child_tag]["@value"], list):
#
#                                 aux[child_tag]["@value"].append(child.text)
#                             else:
#                                 aux[child_tag]["@value"] = [aux[child_tag]["@value"]] + [child.text]
#                         else:
#                             aux[child_tag]["@value"] = child.text
#             else:
#                 aux[child_tag] = {}
#                 for attr in child.attrib:
#                     aux[child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
#                 if isinstance(child.text, str):
#                     if child.text.strip():
#                         aux[child_tag]["@value"] = child.text
#             tree_walk(child, aux_path + [child_tag])
#
#     for child in xml.getchildren():
#         child_tag = re.sub("{.*}", "", child.tag)
#         if child_tag in res_dict[xml_tag]:
#             for attr in child.attrib:
#                 if re.sub("{.*}", "", attr) in res_dict[xml_tag][child_tag]:
#                     if isinstance(res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)], list):
#                         res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)].append(child.attrib[attr])
#                     else:
#                         res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)] = [res_dict[xml_tag][child_tag][
#                                                                                       re.sub("{.*}", "", attr)]] + [
#                                                                                      child.attrib[attr]]
#                 else:
#                     res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
#             if isinstance(child.text, str):
#                 if child.text.strip():
#                     if "@value" in res_dict[xml_tag][child_tag]:
#                         if isinstance(res_dict[xml_tag][child_tag]["@value"], list):
#                             res_dict[xml_tag][child_tag]["@value"].append(child.text)
#                         else:
#                             res_dict[xml_tag][child_tag]["@value"] = [res_dict[xml_tag][child_tag]["@value"]] + [
#                                 child.text]
#                     else:
#                         res_dict[xml_tag][child_tag]["@value"] = child.text
#         else:
#             res_dict[xml_tag][child_tag] = {}
#             for attr in child.attrib:
#                 res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
#             if isinstance(child.text, str):
#                 if child.text.strip():
#                     if "@value" in res_dict[xml_tag][child_tag]:
#                         if isinstance(res_dict[xml_tag][child_tag]["@value"], list):
#                             res_dict[xml_tag][child_tag]["@value"].append(child.text.strip())
#                         else:
#                             res_dict[xml_tag][child_tag]["@value"] = res_dict[xml_tag][child_tag]["@value"] + [
#                                 child.text]
#
#                     else:
#                         res_dict[xml_tag][child_tag]["@value"] = child.text
#         tree_walk(child, path + [child_tag])
#     return res_dict
#
#
# def preprocess_label(label):
#     stop_words = stopwords.words('english')
#     camel_case_split_list = [match.group(0) for match in
#                              re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', label)]
#     list_of_list = [wordninja.split(sub_token) for sub_token in camel_case_split_list]
#     aux = []
#     for list in list_of_list:
#         for token in list:
#             if token not in stop_words:
#                 aux.append(token.lower())
#     return aux
#
#
# def preprocess_label(label):
#     stop_words = stopwords.words('english')
#     camel_case_split_list = [match.group(0) for match in
#                              re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', label)]
#     list_of_list = [wordninja.split(sub_token) for sub_token in camel_case_split_list]
#     aux = []
#     for list in list_of_list:
#         for token in list:
#             if token not in stop_words:
#                 aux.append(token.lower())
#     return aux
#
#
# def get_glove_dict_sum(key_list, glove_model):
#     glove_dict_res = {}
#     key_error = []
#     word_error = []
#     for key in key_list:
#
#         if key in glove_model:
#             glove_dict_res[key] = glove_model[key]
#         else:
#             aux = None
#             for word in preprocess_label(key):
#                 try:
#                     if aux is None:
#                         aux = glove_model[word.lower()]
#                     else:
#                         aux = aux + glove_model[word.lower()]
#                 except KeyError as e:
#                     key_error.append(key)
#                     word_error.append(word)
#                 except Exception as e:
#                     print(e)
#             if aux is not None:
#                 glove_dict_res[key] = aux
#             else:
#                 print(key)
#         pass
#     return glove_dict_res, key_error, word_error
#
#
# def compare_glove_dict_euclid(glove_dict_a, glove_dict_b):
#     compar_dict = {}
#     start = time.time()
#     for odatis_key in glove_dict_a:
#
#         compar_dict[odatis_key] = {}
#         for aeris_key in glove_dict_b:
#             compar_dict[odatis_key][aeris_key] = np.linalg.norm(glove_dict_a[odatis_key] - glove_dict_b[aeris_key])
#     for i in compar_dict:
#         compar_dict[i] = sorted(compar_dict[i].items(), key=lambda item: item[1])[0]
#     end = time.time()
#     print("Time elapsed only for matching for euclidian distance : " + str(end - start))
#     return compar_dict
#
#
# def eval_matches(matches, ground_truth_path):
#     good_match = []
#     bad_match = []
#     ground_truth = pd.read_csv(ground_truth_path).set_index("key")
#     for match_key in matches:
#         if matches[match_key][0] == ground_truth.loc[match_key].match:
#             good_match.append((match_key, matches[match_key]))
#         else:
#             bad_match.append((match_key, matches[match_key]))
#     return good_match, bad_match
#
#
# def pipeline_sum_euclid(key_list_a, key_list_b, glove_model):
#     glove_dict_res_a, _, _ = get_glove_dict_sum(key_list_a, glove_model)
#     glove_dict_res_b, _, _ = get_glove_dict_sum(key_list_b, glove_model)
#     matches = compare_glove_dict_euclid(glove_dict_res_a, glove_dict_res_b)
#     print("Number of matches : " + str(len(matches)))
#     return matches
#
#
# def match_multi_token_label(key_list_a, key_list_b, glove_model):
#     start = time.time()
#     res = {}
#     for key_a in key_list_a:
#         res[key_a] = {}
#         for key_b in key_list_b:
#             res[key_a][key_b] = glove_model.wmdistance(preprocess_label(key_a), preprocess_label(key_b))
#     for key in res:
#         res[key] = sorted(res[key].items(), key=lambda item: item[1])[0]
#     end = time.time()
#     print("Time elapsed only for matching word mover's distance : " + str(end - start))
#     print("Number of matches : " + str(len(res)))
#     return res
#
#
# def pipeline_multi_token_distance(key_list_a, key_list_b, glove_model):
#     # start = time.time()
#
#     matches = match_multi_token_label(key_list_a, key_list_b, glove_model)
#     # end = time.time()
#     # print("Time elapsed only for matching : "+ str(end - start))
#     return matches
#
#
# def select_token(key_list, number, delimiter, split_fun=preprocess_label):
#     return list(map(lambda x: delimiter.join(split_fun(x)[-number:]), key_list))
#
#
# def fun(odatis_key, odatis_number, aeris_key, aeris_number, model):
#     print("A-" + str(aeris_number))
#     return aeris_number, pipeline_sum_euclid(select_token(odatis_key, odatis_number, lambda x: x.split("->")),
#                                              select_token(aeris_key, aeris_number, lambda x: x.split("->")), model)
#
#
# def funB(odatis_key, odatis_number, aeris_key, aeris_number, model):
#     print("B-" + str(aeris_number))
#
#     return aeris_number, pipeline_multi_token_distance(select_token(odatis_key, odatis_number, lambda x: x.split("->")),
#                                                        select_token(aeris_key, aeris_number, lambda x: x.split("->")),
#                                                        model)
#
#
# def expe(glove_vectors, google_glove_vector):
#     print("Start : ")
#     res = {}
#     odatis_key = table2.keys()
#     aeris_key = table1.keys()
#     models_str = ["wikipedia_corpus_word2vec", "google_corpus_word2vec"]
#     res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_sum_euclid(
#         odatis_key, aeris_key, glove_vectors)
#     res["multi_word/wikipedia_corpus_word2vec/odatis_whole/aeris_who0,,le/only_leaf"] = pipeline_multi_token_distance(
#         odatis_key, aeris_key, glove_vectors)
#     res["euclidian_sum_pipeline/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_sum_euclid(
#         odatis_key, aeris_key, google_glove_vector)
#     res["multi_word/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_multi_token_distance(
#         odatis_key, aeris_key, google_glove_vector)
#     # print(len(res["multi_word/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"]))
#     # print(len(res["euclidian_sum_pipeline/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"]))
#     # print(len(res["multi_word/wikipedia_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"]))
#     # print(len(res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"]))
#
#     for index, model in enumerate([glove_vectors, google_glove_vector]):
#         print(models_str[index])
#         for odatis_number in range(1, 12):
#             print(odatis_number)
#
#             results = Parallel(n_jobs=4, backend="multiprocessing")(
#                 map(delayed(fun), [odatis_key, odatis_key, odatis_key, odatis_key],
#                     [odatis_number, odatis_number, odatis_number, odatis_number],
#                     [aeris_key, aeris_key, aeris_key, aeris_key, ], [1, 2, 3, 4], [model, model, model, model]))
#             for i in results:
#                 res["euclidian_sum_pipeline/" + models_str[index] + "/odatis_last_" + str(
#                     odatis_number) + "/aeris_last_" + str(i[0]) + "/strip(\"->\")/only_leaf"] = i[1]
#             resultsB = Parallel(n_jobs=4, backend="multiprocessing")(
#                 map(delayed(funB), [odatis_key, odatis_key, odatis_key, odatis_key],
#                     [odatis_number, odatis_number, odatis_number, odatis_number],
#                     [aeris_key, aeris_key, aeris_key, aeris_key, ], [1, 2, 3, 4], [model, model, model, model]))
#             for i in resultsB:
#                 res["multi_word/" + models_str[index] + "/odatis_last_" + str(odatis_number) + "/aeris_last_" + str(
#                     i[0]) + "/strip(\"->\")/only_leaf"] = i[1]
#     save_file = "Word2vec_expe_" + str(datetime.datetime.now()) + ".dict"
#     with open(save_file, "wb") as f:
#         pickle.dump(res, f)
#     return save_file
#
#
# def expe():
#     import tqdm
#     res = {}
#     res["coma/odatis_whole/aeris_whole/only_leaf"] = valentine_match(table2, table1, Coma(strategy="COMA_OPT"))
#     for b_number in range(1, 12):
#         for a_number in range(1, 5):
#             start = time.time()
#             res["coma/odatis_last_" + str(b_number) + "/aeris_last_" + str(
#                 a_number) + "/strip(\"->\")/only_leaf"] = valentine_match(
#                 pd.DataFrame(columns=list(set(select_token(table2.columns, b_number, lambda x: x.split("->"))))),
#                 pd.DataFrame(columns=list(set(select_token(table1.columns, a_number, lambda x: x.split("->"))))),
#                 Coma(strategy="COMA_OPT"))
#             end = time.time()
#             print("Time elapsed only for matching : " + str(end - start))
#     save_file = "COMA_expe_" + str(datetime.datetime.now()) + ".dict"
#     with open(save_file, "wb") as f:
#         pickle.dump(res, f)
#     return save_file
#
#
# def get_val_from_path(path, dict_to_get, sep="."):
#     x = dict_to_get
#     for sub in path.split(sep):
#         try:
#             x = x[sub]
#         except:
#             return x
#     return x
#
#
# #
#
#
# def get_max_path_length(model_df, delimiter):
#     """
#     Retourne la taille des chemins et le nombre d'occurence de cette taille dans le model
#     :param model_df: list : list of path
#     :return: int : max path size / number of node in the longest path in the model
#     """
#     aux_counter = 0
#     for key_path in model_df:
#         if len(key_path.split(delimiter)) > aux_counter:
#             aux_counter = len(key_path.split(delimiter))
#
#     # res = dict(sorted(res.items()))
#     # return dict(collections.OrderedDict(sorted(res.items()))
#     return aux_counter
#
#
# if __name__ == '__main__':
#     path_to_mapping = "mappings/"
#     path_to_models = "models/"
#     path_to_res = "results/"
#
#     print("Don't forget to start mongodb docker container with port 27017 open on localhost.")
#     print("Starting model extraction from files in..." + path_to_models)
#     mongo_client = MongoClient("localhost:27017")
#     base_path = "."
#     model_dict = {}
#     model_examples_folder = list(os.walk(path_to_models))[0][1]
#     # for model_name in model_examples_folder:
#     #     print(model_name)
#     #     file_list = []
#     #     mongodb_coll_var = mongo_client[model_name].interop_metadata
#     #
#     #     for i in os.walk(path_to_models+model_name):
#     #         for j in i[2]:
#     #             if j.endswith(".json"):
#     #                 file_path = os.path.join(i[0], j)
#     #                 file_list.append((os.path.join(i[0], j), model_name, "json"))
#     #                 read_preprocess_insert_in_mongodb_json(fkey_listile_path, mongodb_coll=mongodb_coll_var)
#     #             if j.endswith(".xml"):
#     #                 try:
#     #                     file_list.append((os.path.join(i[0], j), model_name, "xml"))
#     #                     file = etree.parse(open(os.path.join(i[0], j)),
#     #                                        parser=etree.XMLParser(ns_clean=True, remove_comments=True,
#     #                                                               recover=True)).getroot()
#     #                     res = XML_to_dict(file)
#     #                     read_preprocess_insert_in_mongodb_json(res, mongodb_coll=mongodb_coll_var, fp_is_dict=True)
#     #                 except Exception as e:
#     #                     print(os.path.join(i[0], j))
#     #                     print(e)
#     #     model_dict[model_name] = file_list
#     #
#     # # pd.DataFrame(model_dict["FHIR"],columns=["Filepath","Model","File extension"])
#     #
#     # # model_dict
#     # for model_name in model_dict:
#     #     mongodb_coll_var = mongo_client[model_name].interop_metadata
#     #     model_key_set = []
#     #     model_key_set = set(model_key_set)
#     #     docs = mongodb_coll_var.find()
#     #     for doc in docs:
#     #         model_key_set = model_key_set.union(set(get_all_keys(doc, separator=".", key_list=[])))
#     #     distinct_keys = {}
#     #     for key in model_key_set:
#     #         if key != "_id":
#     #             value_filled = False
#     #             value = mongodb_coll_var.distinct(key)
#     #             for obj in value:
#     #                 if isinstance(obj, type({})):
#     #                     value_filled = True
#     #                     break
#     #             if not value_filled:
#     #                 distinct_keys[key] = {
#     #                     "count": mongodb_coll_var.count_documents({key: {"$exists": True}}),
#     #                     "values": value
#     #                 }
#     #     pd.DataFrame(distinct_keys).transpose().to_csv(path_to_models+model_name + "_model.csv")
#
#     #
#     #
#     # mongo_client = MongoClient("localhost:27017")
#     # model_dict = {}
#     # model_examples_folder = list(os.walk(path_to_models))[0][1]
#     # print(model_examples_folder)
#     # for model_name in model_examples_folder:
#     #     print(model_name)
#     #     file_list = []
#     #     mongodb_coll_var = mongo_client[model_name].interop_metadata
#     #     for i in os.walk(path_to_models+model_name):
#     #         for j in i[2]:
#     #             if j.endswith(".json"):
#     #                 file_path = os.path.join(i[0], j)
#     #                 file_list.append((os.path.join(i[0], j), model_name, "json"))
#     #                 read_preprocess_insert_in_mongodb_json(file_path, mongodb_coll=mongodb_coll_var)
#     #             if j.endswith(".xml"):
#     #                 try:
#     #                     file_list.append((os.path.join(i[0], j), model_name, "xml"))
#     #                     file = etree.parse(open(os.path.join(i[0], j)),
#     #                                        parser=etree.XMLParser(ns_clean=True, remove_comments=True,
#     #                                                               recover=True)).getroot()
#     #                     res = XML_to_dict(file)
#     #                     # print(res)
#     #                     read_preprocess_insert_in_mongodb_json(res, mongodb_coll=mongodb_coll_var, fp_is_dict=True)
#     #                 except Exception as e:
#     #                     print(os.path.join(i[0], j))
#     #                     print(e)
#     #     model_dict[model_name] = file_list
#     #
#     # for model_name in model_dict:
#     #     mongodb_coll_var = mongo_client[model_name].interop_metadata
#     #     model_key_set = []
#     #     model_key_set = set(model_key_set)
#     #     docs = list(mongodb_coll_var.find())
#     #     for doc in docs:
#     #         model_key_set = model_key_set.union(set(get_all_keys(doc, separator=".", key_list=[])))
#     #     distinct_keys = {}
#     #     for key in model_key_set:
#     #         distinct_keys[key]={"count":0,"values":[]}
#     #         # if key != "_id":
#     #         value_filled = False
#     #         list_doc = list(mongodb_coll_var.find({key: {"$exists": True}},{"_id":0,key:1}))
#     #         values = []
#     #         for doc in list_doc:
#     #             try:
#     #                 values.append(get_val_from_path(key, doc))
#     #             except:
#     #                 continue
#     #         distinct_keys[key] = {
#     #             "count": len(list_doc),
#     #             "values": values
#     # print("Extract mapping from cda2r4.")
#     #
#     # res = []
#     # num_files = 0
#     # for enumerator, folder in enumerate(list(os.walk(path_to_mapping+"cda2r4/source"))):
#     #     print(str(enumerator) + "/" + (str(len(list(os.walk(path_to_mapping+"cda2r4/source"))))))
#     #     if folder[2]:
#     #         num_files += len(folder[2])
#     #         for index, file_name in enumerate(folder[2]):
#     #             if index == int(len(folder[2])/2) or index == 0 or index == int(len(folder[2])/1.1):
#     #                 print(str(index) + "/" + str(len(folder[2])))
#     #             var_input = folder[0] + "/" + file_name
#     #             var_output = folder[0].replace("source", "output") + "/FHIR_" + file_name.replace(".xml", ".json")
#     #             input_xml = XML_to_dict(etree.parse(open(var_input),
#     #                                                 parser=etree.XMLParser(ns_clean=True, remove_comments=True,
#     #                                                                        recover=True)).getroot())
#     #             output_json = json.load(open(var_output))
#     #             mapping_list = []
#     #             map_two_dict(output_json, input_xml, mapping_list)
#     #             for mapping in mapping_list:
#     #                 # Exclusion des multi mapping (1:n)
#     #                 if len(mapping[1]) == 1:
#     #                     remove_number = []
#     #                     for tag in mapping[0].split("."):
#     #                         if not tag.isdigit():
#     #                             remove_number.append(tag)
#     #                     # mapping[1][0] est une liste de taille 1, on prend juste le contenu
#     #                     res.append((".".join(remove_number), mapping[1][0]))
#     #
#     # pd.DataFrame(set(res), columns=["key", "match"]).set_index("key").to_csv(path_to_mapping+"FHIR_to_C-CDA_._delimiter.csv")
#     # print(str(num_files) + " files processed.")
#     #         }
#     #     pd.DataFrame(distinct_keys).transpose().to_csv(path_to_models+model_name + "_model.csv")
#     #
#     # print("Extract mapping from cda2r4.")
#     #
#     # res = []
#     # num_files = 0
#     # for enumerator, folder in enumerate(list(os.walk(path_to_mapping+"cda2r4/source"))):
#     #     print(str(enumerator) + "/" + (str(len(list(os.walk(path_to_mapping+"cda2r4/source"))))))
#     #     if folder[2]:
#     #         num_files += len(folder[2])
#     #         for index, file_name in enumerate(folder[2]):
#     #             if index == int(len(folder[2])/2) or index == 0 or index == int(len(folder[2])/1.1):
#     #                 print(str(index) + "/" + str(len(folder[2])))
#     #             var_input = folder[0] + "/" + file_name
#     #             var_output = folder[0].replace("source", "output") + "/FHIR_" + file_name.replace(".xml", ".json")
#     #             input_xml = XML_to_dict(etree.parse(open(var_input),
#     #                                                 parser=etree.XMLParser(ns_clean=True, remove_comments=True,
#     #                                                                        recover=True)).getroot())
#     #             output_json = json.load(open(var_output))
#     #             mapping_list = []
#     #             map_two_dict(output_json, input_xml, mapping_list)
#     #             for mapping in mapping_list:
#     #                 # Exclusion des multi mapping (1:n)
#     #                 if len(mapping[1]) == 1:
#     #                     remove_number = []
#     #                     for tag in mapping[0].split("."):
#     #                         if not tag.isdigit():
#     #                             remove_number.append(tag)
#     #                     # mapping[1][0] est une liste de taille 1, on prend juste le contenu
#     #                     res.append((".".join(remove_number), mapping[1][0]))
#     #
#     # pd.DataFrame(set(res), columns=["key", "match"]).set_index("key").to_csv(path_to_mapping+"FHIR_to_C-CDA_._delimiter.csv")
#     # print(str(num_files) + " files processed.")
#
#     # print("Starting matching with COMA..")
#     # onlyfiles = [f for f in os.listdir(path_to_mapping) if
#     #              os.path.isfile(os.path.join(path_to_mapping, f)) and f.endswith(".csv")]
#     # print(onlyfiles)
#     # for ground_truth in onlyfiles:
#     #     print(ground_truth)
#     #     parameters = ground_truth.split("_")
#     #     modelA_name = parameters[0]
#     #     modelB_name = parameters[2]
#     #     delimiter = parameters[3]
#     #     modelA = pd.read_csv(path_to_models + modelA_name + "_model.csv").set_index("Unnamed: 0").transpose()
#     #     modelB = pd.read_csv(path_to_models + modelB_name + "_model.csv").set_index("Unnamed: 0").transpose()
#     #     if delimiter == "arrow":
#     #         modelA = modelA.rename(columns=lambda x: x.replace(".", "->"))
#     #         modelB = modelB.rename(columns=lambda x: x.replace(".", "->"))
#     #     print(modelA)
#     #     print(modelB)
#     #     print("Begginning of matching with Coma(strategy=\"COMA_OPT\")")
#     #     start = time.time()
#     #     matchs_COMA_A = valentine_match(modelA, modelB, Coma(strategy="COMA_OPT"))
#     #     print(str(len(matchs_COMA_A)) + " matches done.")
#     #     end = time.time()
#     #     print("Time elapsed only for matching : " + str(end - start))
#     #
#     #     # good_match = []
#     #     #
#     #     # ground_truth_val = pd.read_csv(path_to_mapping+ground_truth).set_index("key")
#     #     # for match in matchs_COMA_A:
#     #     #     table2_key = match[0][1]
#     #     #     table1_match = match[1][1]
#     #     #     ground_truth_value = ground_truth.loc()[table2_key].match
#     #     #     if table1_match == ground_truth_value:
#     #     #         good_match.append(match)
#     #     # print(str(len(good_match)) + " good matches")
#     #     # print(str(len(matchs_COMA_A) - len(good_match)) + " bad matches")
#     #     # print(str(len(matchs_COMA_A)) + " total matches")
#     #     # print("Precision = " + str(len(good_match) / len(matchs_COMA_A)))
#     #     open(ground_truth+".txt","w").write(str(matchs_COMA_A))
#     print("Start matching with Word2Vec...")
#
#     # ODATIS = pd.read_csv("ground_truth_only_leaf.csv")
#     # AERIS = pd.read_csv("aeris_only_leaf.csv")
#     # odatis_key = ODATIS.key
#     # aeris_key = AERIS.key
#     # print(modelA)
#     # for key in aeris_key:
#     #     print(key)
#     print("Loading glove_vectors..")  # print(onlyfiles)
#
#     # glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
#     onlyfiles = [f for f in os.listdir(path_to_mapping) if
#                  os.path.isfile(os.path.join(path_to_mapping, f)) and f.endswith(".csv")]
#
#     res_files_list = [f for f in os.listdir(path_to_res) if
#                       os.path.isfile(os.path.join(path_to_res, f)) and f.endswith(".txt")]
#     # print(glove_vectors)
#     print("Starting matching...")
#     # for ground_truth in onlyfiles:
#     #     path_to_mapping = "mappings/"
#     #     path_to_models = "models/"
#     #     print("Don't forget to start mongodb docker container with port 27017 open on localhost.")
#     #     print("Starting model extraction from files in..." + path_to_models)
#     #     mongo_client = MongoClient("localhost:27017")
#     #     base_path = "."
#     #     model_dict = {}
#     #     model_examples_folder = list(os.walk(path_to_models))[0][1]
#     #     print("Starting matching...")
#     #     onlyfiles = [f for f in os.listdir(path_to_mapping) if
#     #                  os.path.isfile(os.path.join(path_to_mapping, f)) and f.endswith(".csv")]
#
#     # for ground_truth in onlyfiles:
#     #     print(ground_truth)
#     #     res = {}
#     #     parameters = ground_truth.split("_")
#     #     modelA_name = parameters[0]
#     #     modelB_name = parameters[2]
#     #     delimiter = parameters[3]
#     #     modelA = pd.read_csv(path_to_models + modelA_name + "_model.csv").set_index("Unnamed: 0").transpose()
#     #     modelB = pd.read_csv(path_to_models + modelB_name + "_model.csv").set_index("Unnamed: 0").transpose()
#     #
#     #     modelA = modelA.rename(columns=lambda x: x.replace(".", delimiter))
#     #     modelB = modelB.rename(columns=lambda x: x.replace(".", delimiter))
#     #     # print(modelA.columns)
#     #     # print(modelB.columns)
#     #
#     #     glove_dict_res = {}
#     #     key_error = []
#     #     word_error = []
#     #
#     #     # for key in tqdm.tqdm(modelA.columns):
#     #     #     pass
#     #     # if key in glove_model:
#     #     #     glove_dict_res[key] = glove_model[key.lower()]
#     #     # else:
#     #     #     aux = None
#     #     #     for word in preprocess_label(key):
#     #     #         try:
#     #     #             if aux is None:
#     #     #                 aux = glove_model[word.lower()]
#     #     #             else:
#     #     #                 aux = aux + glove_model[word.lower()]
#     #     #         except KeyError as e:
#     #     #             key_error.append(key)
#     #     #             word_error.append(word)
#     #     #         except Exception as e:
#     #     #             print(e)
#     #     #     if aux is not None:
#     #     #         glove_dict_res[key] = aux
#     #     #     else:
#     #     #         print(key)
#     #     # pass
#     #     for sub_path_A_size in range(get_max_path_length(modelA.columns, delimiter)):
#     #         for sub_path_B_size in range(get_max_path_length(modelB.columns, delimiter)):
#     #             # print()
#     #             if "Word2Vec_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
#     #                  + str(sub_path_B_size) + "-sized_path.dict.txt" not in res_files_list:
#     #                 res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] = {}
#     #                 start = time.time()
#     #                 res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name]["results"] = \
#     #                     pipeline_sum_euclid(list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
#     #                                                               lambda x: x.split(delimiter)))),
#     #                                         list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
#     #                                                               lambda x: x.split(delimiter)))),
#     #                                         glove_vectors)
#     #                 end = time.time()
#     #                 print("Time elapsed only for matching with euclidian distance for "+ modelA_name + " with "
#     #                       + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
#     #                       + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
#     #                 res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] \
#     #                     ["processing_time"] = str(end - start)
#     #                 start = time.time()
#     #
#     #                 res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] = {}
#     #
#     #                 res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name]["results"] = \
#     #                     pipeline_multi_token_distance(
#     #                         list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
#     #                                               lambda x: x.split(delimiter)))),
#     #                         list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
#     #                                               lambda x: x.split(delimiter)))),
#     #                         glove_vectors)
#     #                 end = time.time()
#     #                 print("Time elapsed only for matching with word mover's distance for "+ modelA_name + " with "
#     #                       + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
#     #                       + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
#     #                 res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] \
#     #                     ["processing_time"] = str(end - start)
#     #                 # res[
#     #                 #     "euclidian_sum_pipeline/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_sum_euclid(
#     #                 #     modelA.columns, aeris_key, google_glove_vector)
#     #                 # res["multi_word/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_multi_token_distance(
#     #                 #     modelA.columns, aeris_key, google_glove_vector)
#     #                 #
#     #                 # print(res)
#     #                 with open(path_to_res+"Word2Vec_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"
#     #                      + str(sub_path_B_size) + "-sized_path.dict.txt", "wb") as f:
#     #                     pickle.dump(res, f)
#
#     for ground_truth in onlyfiles:
#         print(ground_truth)
#         res = {}
#         parameters = ground_truth.split("_")
#         modelA_name = parameters[0]
#         modelB_name = parameters[2]
#         delimiter = parameters[3]
#         modelA = pd.read_csv(path_to_models + modelA_name + "_model.csv").set_index("Unnamed: 0").transpose()
#         modelB = pd.read_csv(path_to_models + modelB_name + "_model.csv").set_index("Unnamed: 0").transpose()
#
#         modelA = modelA.rename(columns=lambda x: x.replace(".", delimiter))
#         modelB = modelB.rename(columns=lambda x: x.replace(".", delimiter))
#         # print(modelA.columns)
#         # print(modelB.columns)
#
#         glove_dict_res = {}
#         key_error = []
#         word_error = []
#
#         # for key in tqdm.tqdm(modelA.columns):
#         #     pass
#         # if key in glove_model:
#         #     glove_dict_res[key] = glove_model[key.lower()]
#         # else:
#         #     aux = None
#         #     for word in preprocess_label(key):
#         #         try:
#         #             if aux is None:
#         #                 aux = glove_model[word.lower()]
#         #             else:
#         #                 aux = aux + glove_model[word.lower()]
#         #         except KeyError as e:
#         #             key_error.append(key)
#         #             word_error.append(word)
#         #         except Exception as e:
#         #             print(e)
#         #     if aux is not None:
#         #         glove_dict_res[key] = aux
#         #     else:
#         #         print(key)
#         # pass
#         res_full={}
#         if "COMA_FHIR_6-sizedpath_C-CDA_19-sized_path.dict.txt" not in res_files_list:
#             print("Beggin matching for : COMA_FHIR_full-sizedpath_C-CDA_full-sized_path.dict.txt")
#             res_full["COMA/" + modelA_name + "/" + modelB_name] = {}
#             start = time.time()
#
#             # valentine_match(modelA, modelB, Coma(strategy="COMA_OPT"))
#             res_full["COMA/" + modelA_name + "/" + modelB_name]["results"] = \
#                 valentine_match(modelA, modelB, Coma(strategy="COMA_OPT"))
#             end = time.time()
#             print("Time elapsed only for matching with euclidian distance for " + modelA_name + " and " + modelB_name + " : " + str(end - start))
#             res_full["COMA/" + modelA_name + "/" + modelB_name] \
#                 ["processing_time"] = str(end - start)
#             start = time.time()
#             with open(
#                     path_to_res + "COMA_" + modelA_name + "_6-sizedpath_" + modelB_name + "_19-sized_path.dict.txt", "wb") as f:
#                 pickle.dump(res_full, f)
#
#         for sub_path_A_size in range(get_max_path_length(modelA.columns, delimiter)):
#             for sub_path_B_size in range(get_max_path_length(modelB.columns, delimiter)):
#                 # print()
#
#                 if "COMA_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
#                      + str(sub_path_B_size) + "-sized_path.dict.txt" not in res_files_list:
#                     print("Beggin matching for : " + "COMA"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
#                      + str(sub_path_B_size) + "-sized_path.dict.txt" )
#                     res["COMA/" + modelA_name + "/" + modelB_name] = {}
#                     start = time.time()
#
#                     # valentine_match(modelA, modelB, Coma(strategy="COMA_OPT"))
#                     res["COMA/" + modelA_name + "/" + modelB_name]["results"] = \
#                         valentine_match(pd.DataFrame(columns=(list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
#                                                                   lambda x: x.split(delimiter)))))),
#                                             pd.DataFrame(columns=list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
#                                                                   lambda x: x.split(delimiter))))), Coma(strategy="COMA_OPT"))
#                     end = time.time()
#                     print("Time elapsed only for matching with euclidian distance for "+ modelA_name + " with "
#                           + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
#                           + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
#                     res["COMA/" + modelA_name + "/" + modelB_name] \
#                         ["processing_time"] = str(end - start)
#                     start = time.time()
#
#                     # res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] = {}
#                     #
#                     # res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name]["results"] = \
#                     #     pipeline_multi_token_distance(
#                     #         list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
#                     #                               lambda x: x.split(delimiter)))),
#                     #         list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
#                     #                               lambda x: x.split(delimiter)))),
#                     #         glove_vectors)
#                     # end = time.time()
#                     # print("Time elapsed only for matching with word mover's distance for "+ modelA_name + " with "
#                     #       + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
#                     #       + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
#                     # res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] \
#                     #     ["processing_time"] = str(end - start)
#                     # res[
#                     #     "euclidian_sum_pipeline/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_sum_euclid(
#                     #     modelA.columns, aeris_key, google_glove_vector)
#                     # res["multi_word/google_corpus_word2vec/odatis_whole/aeris_whole/only_leaf"] = pipeline_multi_token_distance(
#                     #     modelA.columns, aeris_key, google_glove_vector)
#                     #
#                     # print(res)
#                     with open(path_to_res+"COMA_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"
#                          + str(sub_path_B_size) + "-sized_path.dict.txt", "wb") as f:
#                         pickle.dump(res, f)
import pprint
from os import listdir
from os.path import isfile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dateparser
import re
import pickle

import tqdm
from pymongo import MongoClient
import json
import os
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid
from lxml import etree
import datetime
import gensim.downloader
import wordninja
from nltk import download
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import time


def check_value_in_dict(val, dict_to_check, res_list, path=None):
    for i in dict_to_check:
        if path is None:
            aux_path = i
        else:
            aux_path = path + "." + i
        # if isinstance(val, list):
        #     if dict_to_check[i] == val or dict_to_check[i] == val:
        #         pass
        if dict_to_check[i] == val:
            res_list.append(aux_path)
        if isinstance(dict_to_check[i], dict):
            check_value_in_dict(val, dict_to_check[i], res_list, aux_path)


def map_two_dict(dict_a, dict_b, res_tuple_list, path=None):
    for path_iterator_var in dict_a:
        if isinstance(dict_a[path_iterator_var], dict):
            if path is None:
                path_aux = path_iterator_var
            else:
                path_aux = path + "." + path_iterator_var

            map_two_dict(dict_a[path_iterator_var], dict_b, res_tuple_list, path_aux)
        else:
            if isinstance(dict_a[path_iterator_var], list):
                contains_dict = False
                for index, element in enumerate(dict_a[path_iterator_var]):
                    if isinstance(element, dict):
                        contains_dict = True
                        if path is None:
                            path_aux = path_iterator_var
                        else:
                            path_aux = path + "." + str(index) + "." + path_iterator_var
                        map_two_dict(element, dict_b, res_tuple_list, path_aux)
                if not contains_dict:
                    for index, element in enumerate(dict_a[path_iterator_var]):
                        list_aux = []
                        check_value_in_dict(element, dict_b, list_aux)
                        if list_aux:
                            if path is None:
                                res_tuple_list.append((path_iterator_var, list_aux))
                            else:
                                res_tuple_list.append((path + "." + str(index) + "." + path_iterator_var, list_aux))
            list_aux = []
            check_value_in_dict(dict_a[path_iterator_var], dict_b, list_aux)
            if list_aux:
                if path is None:
                    res_tuple_list.append((path_iterator_var, list_aux))
                else:
                    res_tuple_list.append((path + "." + path_iterator_var, list_aux))


def get_all_keys(doc, main_key=None, separator=".", key_list=[], first_call=True):
    '''
    Get all key and sub key of a document, sub key are constructed with a separator defined as a parameter.
    :param doc:
    :param main_key:
    :param separator:
    :param first_call
    :return:
    '''
    if isinstance(doc, type({})):

        for key in doc.keys():
            if main_key is None:
                key_list.append(key)
                get_all_keys(doc[key], main_key=key, separator=separator, key_list=key_list, first_call=False)
            else:
                key_list.append(main_key + separator + key)
                get_all_keys(doc[key], main_key=main_key + separator + key, separator=separator, key_list=key_list,
                             first_call=False)
    if isinstance(doc, type([])):
        for obj in doc:
            if isinstance(obj, type({})):
                for key in obj.keys():
                    if main_key is None:
                        key_list.append(key)
                        get_all_keys(obj[key], main_key=key, separator=separator, key_list=key_list, first_call=False)
                    else:
                        key_list.append(main_key + separator + key)
                        get_all_keys(obj[key], main_key=main_key + separator + key, separator=separator,
                                     key_list=key_list, first_call=False)

    return key_list


def remove_id_in_json(f_json):
    """
    As mongoDB id or _id is reserved keyword, this function modify any field that is "id" or "_id" to add "doc" before.
    :param f_json: dict containing a parsed json
    :return: dict : f_json with a modified "id" or "_id" key if there was
    """
    key_list = ["id", "_id"]
    if index_key_list := [index for index, key_is_present in enumerate([key in f_json for key in key_list]) if
                          key_is_present]:
        for index in index_key_list:
            f_json["doc_" + key_list[index]] = f_json.pop(key_list[index])
    return f_json


def format_date_json(doc):
    if type(doc) is dict:
        for key in doc.keys():
            if (type(doc[key]) is str) and (date := dateparser.parse(doc[key])) is not None:
                doc[key] = date
            format_date_json(doc[key])
    if type(doc) is list:
        for object in doc:
            format_date_json(object)

    return doc


def read_preprocess_insert_in_mongodb_json(fp, mongodb_coll=None, fp_is_dict=False):
    """
    Read, remove any incompatible "id" key and insert the JSON in a collection in a mongodb database
    :param fp: str : file_path to a JSON to read
    :param mongodb_coll: MongoClient.database.collection : A collection in which insert files
    :return: None
    """
    if mongodb_coll is None:
        mongodb_coll = MongoClient("localhost:27017").no_model_name.interop_metadata
    try:
        # mongodb_coll.insert_one(format_date_json(remove_id_in_json(json.load(open(fp)))))
        # Error seens in data formatting
        if fp_is_dict:
            mongodb_coll.insert_one((remove_id_in_json(fp)))
        else:
            mongodb_coll.insert_one((remove_id_in_json(json.load(open(fp)))))
        # print(fp + " has been inserted successfully.")
    except Exception as exce:
        print("Insertion has not been successfully done. Logs : " + str(exce))


def dont_contains_dict(liste):
    for elem in liste:
        # print(elem)
        if type(elem) is dict:
            return False
    return True


# Transform a date to standard format
def format_date(date):
    # Transform a string date into a standard format by trying each
    # date format. If you want to add a format, add a try/except in the
    # last except
    # date : str : the date to transform
    # return : m : timedata : format is YYYY-MM-DD HH:MM:SS
    date_str = date
    #
    date_str = date_str.replace("st", "").replace("th", "") \
        .replace("nd", "").replace("rd", "").replace(" Augu ", " Aug ")
    m = None
    sep_list = [".", "/", "-", "_", " ", ":"]
    for date_sep in sep_list:
        try:
            m = datetime.datetime.strptime(date_str, "%d" + date_sep + "%B" + date_sep + "%Y")
            break
        except ValueError:
            try:
                m = datetime.datetime.strptime(date_str, "%d" + date_sep + "%b" + date_sep + "%Y")
                break
            except ValueError:
                try:
                    m = datetime.datetime.strptime(date_str, "%Y" + date_sep + "%m" + date_sep + "%d")
                    break
                except ValueError:
                    try:
                        m = datetime.datetime.strptime(date_str,
                                                       "%d" + date_sep + "%m" + date_sep + "%Y")
                        break
                    except ValueError:
                        for hour_sep in sep_list:
                            try:
                                m = datetime.datetime \
                                    .strptime(date_str,
                                              "%d" + date_sep + "%m" + date_sep + "%Y %H" + hour_sep + "%M" + hour_sep + "%S")
                                break
                            except ValueError:
                                try:
                                    m = datetime.datetime \
                                        .strptime(date_str,
                                                  "%Y" + date_sep + "%m" + date_sep + "%d %H" + hour_sep + "%M" + hour_sep + "%S")
                                    break
                                except ValueError:
                                    # HERE ADD A FORMAT TO CHECK
                                    # print("Format not recognised. \nConsider "
                                    #       "adding a date format "
                                    #       "in the function \"format_date\".")
                                    pass

    return m


def from_keyset_to_csv(key_set, separator="."):
    '''
    :param key_set:
    :param separator:
    :return:
    '''
    color = {"0": "#F3722C", "1": "#F8961E", "2": "#F9C74F", "3": "#90BE6D", "4": "#43AA8B", "5": "#4D908E",
             "6": "#577590", "7": "#277DA1", "8": "#bdd5ea", "9": "#e3e2e3", "10": "#ffffff", "11": "#ffffff"}
    csv_list = [("ROOT_NODE", "", -1, "#F94144")]
    for key_concat in key_set:
        key_split = key_concat.split(separator)
        for index in range(len(key_split)):
            if index < 9:
                if len(key_split) == 1:
                    csv_list.append((separator.join(key_split[:index + 1]), "ROOT_NODE", index, color[str(index)]))
                    break
                if index == len(key_split) - 1:
                    break
                csv_list.append((
                    separator.join(key_split[:index + 2]), separator.join(key_split[:index + 1]), index + 1,
                    color[str(index + 1)]))

            else:
                if index == len(key_split) - 1:
                    break
                csv_list.append((
                    separator.join(key_split[:index + 2]), separator.join(key_split[:index + 1]), index + 1,
                    "#ffffff"))
    return (["key", "mother_key", "level", "color"], set(csv_list))


def get_leaf(node_list):
    aux_json = {}

    def merge(d1, d2):
        for k in d2:
            if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
                merge(d1[k], d2[k])
            else:
                d1[k] = d2[k]

    def create_dict_from_label(token_list):
        aux = {}
        token = token_list.pop(0)
        if token_list:
            aux[token] = create_dict_from_label(token_list)
        else:
            aux[token] = {}
        return aux

    for i in node_list:
        merge(aux_json, create_dict_from_label(i.split("->")))

    def paths(tree, cur=""):
        if not tree:
            yield cur
        else:
            for n, s in tree.items():
                if cur == "":
                    for path in paths(s, n):
                        yield path
                else:
                    for path in paths(s, cur + "->" + n):
                        yield path

    return list(paths(aux_json))


def XML_to_dict(xml):
    xml_tag = re.sub("{.*}", "", xml.tag)
    res_dict = {xml_tag: {}}
    path = [xml_tag]

    def tree_walk(xml, path):
        aux_path = path
        for child in xml.getchildren():
            child_tag = re.sub("{.*}", "", child.tag)
            aux = res_dict
            for i in path:
                aux = aux[i]
            if child_tag in aux:
                for attr in child.attrib:

                    if re.sub("{.*}", "", attr) in aux[child_tag]:

                        if isinstance(aux[child_tag][re.sub("{.*}", "", attr)], list):
                            aux[child_tag][re.sub("{.*}", "", attr)].append(child.attrib[attr])
                        else:
                            aux[child_tag][re.sub("{.*}", "", attr)] = [aux[child_tag][re.sub("{.*}", "", attr)]] + [
                                child.attrib[attr]]
                    else:
                        aux[child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
                if isinstance(child.text, str):
                    if child.text.strip():
                        if "@value" in aux[child_tag]:
                            if isinstance(aux[child_tag]["@value"], list):

                                aux[child_tag]["@value"].append(child.text)
                            else:
                                aux[child_tag]["@value"] = [aux[child_tag]["@value"]] + [child.text]
                        else:
                            aux[child_tag]["@value"] = child.text
            else:
                aux[child_tag] = {}
                for attr in child.attrib:
                    aux[child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
                if isinstance(child.text, str):
                    if child.text.strip():
                        aux[child_tag]["@value"] = child.text
            tree_walk(child, aux_path + [child_tag])

    for child in xml.getchildren():
        child_tag = re.sub("{.*}", "", child.tag)
        if child_tag in res_dict[xml_tag]:
            for attr in child.attrib:
                if re.sub("{.*}", "", attr) in res_dict[xml_tag][child_tag]:
                    if isinstance(res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)], list):
                        res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)].append(child.attrib[attr])
                    else:
                        res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)] = [res_dict[xml_tag][child_tag][
                                                                                      re.sub("{.*}", "", attr)]] + [
                                                                                     child.attrib[attr]]
                else:
                    res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
            if isinstance(child.text, str):
                if child.text.strip():
                    if "@value" in res_dict[xml_tag][child_tag]:
                        if isinstance(res_dict[xml_tag][child_tag]["@value"], list):
                            res_dict[xml_tag][child_tag]["@value"].append(child.text)
                        else:
                            res_dict[xml_tag][child_tag]["@value"] = [res_dict[xml_tag][child_tag]["@value"]] + [
                                child.text]
                    else:
                        res_dict[xml_tag][child_tag]["@value"] = child.text
        else:
            res_dict[xml_tag][child_tag] = {}
            for attr in child.attrib:
                res_dict[xml_tag][child_tag][re.sub("{.*}", "", attr)] = child.attrib[attr]
            if isinstance(child.text, str):
                if child.text.strip():
                    if "@value" in res_dict[xml_tag][child_tag]:
                        if isinstance(res_dict[xml_tag][child_tag]["@value"], list):
                            res_dict[xml_tag][child_tag]["@value"].append(child.text.strip())
                        else:
                            res_dict[xml_tag][child_tag]["@value"] = res_dict[xml_tag][child_tag]["@value"] + [
                                child.text]

                    else:
                        res_dict[xml_tag][child_tag]["@value"] = child.text
        tree_walk(child, path + [child_tag])
    return res_dict


def preprocess_label(label):
    stop_words = stopwords.words('english')
    camel_case_split_list = [match.group(0) for match in
                             re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', label)]
    list_of_list = [wordninja.split(sub_token) for sub_token in camel_case_split_list]
    aux = []
    for list in list_of_list:
        for token in list:
            if token not in stop_words:
                aux.append(token.lower())
    return aux


def preprocess_label(label):
    stop_words = stopwords.words('english')
    camel_case_split_list = [match.group(0) for match in
                             re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', label)]
    list_of_list = [wordninja.split(sub_token) for sub_token in camel_case_split_list]
    aux = []
    for list in list_of_list:
        for token in list:
            if token not in stop_words:
                aux.append(token.lower())
    return aux


def get_glove_dict_sum(key_list, glove_model):
    glove_dict_res = {}
    key_error = []
    word_error = []
    for key in key_list:

        if key in glove_model:
            glove_dict_res[key] = glove_model[key]
        else:
            aux = None
            for word in preprocess_label(key):
                try:
                    if aux is None:
                        aux = glove_model[word.lower()]
                    else:
                        aux = aux + glove_model[word.lower()]
                except KeyError as e:
                    key_error.append(key)
                    word_error.append(word)
                except Exception as e:
                    print(e)
            if aux is not None:
                glove_dict_res[key] = aux
            else:
                print(key)
        pass
    return glove_dict_res, key_error, word_error


def compare_glove_dict_euclid(glove_dict_a, glove_dict_b):
    compar_dict = {}
    start = time.time()
    for odatis_key in glove_dict_a:

        compar_dict[odatis_key] = {}
        for aeris_key in glove_dict_b:
            compar_dict[odatis_key][aeris_key] = np.linalg.norm(glove_dict_a[odatis_key] - glove_dict_b[aeris_key])
    for i in compar_dict:
        compar_dict[i] = sorted(compar_dict[i].items(), key=lambda item: item[1])[0]
    end = time.time()
    print("Time elapsed only for matching for euclidian distance : " + str(end - start))
    return compar_dict


def eval_matches(matches, ground_truth_path):
    good_match = []
    bad_match = []
    ground_truth = pd.read_csv(ground_truth_path).set_index("key")
    for match_key in matches:
        if matches[match_key][0] == ground_truth.loc[match_key].match:
            good_match.append((match_key, matches[match_key]))
        else:
            bad_match.append((match_key, matches[match_key]))
    return good_match, bad_match


def pipeline_sum_euclid(key_list_a, key_list_b, glove_model):
    glove_dict_res_a, _, _ = get_glove_dict_sum(key_list_a, glove_model)
    glove_dict_res_b, _, _ = get_glove_dict_sum(key_list_b, glove_model)
    matches = compare_glove_dict_euclid(glove_dict_res_a, glove_dict_res_b)
    print("Number of matches : " + str(len(matches)))
    return matches


def match_multi_token_label(key_list_a, key_list_b, glove_model):
    start = time.time()
    res = {}
    for key_a in key_list_a:
        res[key_a] = {}
        for key_b in key_list_b:
            res[key_a][key_b] = glove_model.wmdistance(preprocess_label(key_a), preprocess_label(key_b))
    for key in res:
        res[key] = sorted(res[key].items(), key=lambda item: item[1])[0]
    end = time.time()
    print("Time elapsed only for matching word mover's distance : " + str(end - start))
    print("Number of matches : " + str(len(res)))
    return res


def pipeline_multi_token_distance(key_list_a, key_list_b, glove_model):
    # start = time.time()

    matches = match_multi_token_label(key_list_a, key_list_b, glove_model)
    # end = time.time()
    # print("Time elapsed only for matching : "+ str(end - start))
    return matches


def select_token(key_list, number, delimiter, split_fun=preprocess_label):
    return list(map(lambda x: delimiter.join(split_fun(x)[-number:]), key_list))


def fun(odatis_key, odatis_number, aeris_key, aeris_number, model):
    print("A-" + str(aeris_number))
    return aeris_number, pipeline_sum_euclid(select_token(odatis_key, odatis_number, lambda x: x.split("->")),
                                             select_token(aeris_key, aeris_number, lambda x: x.split("->")), model)


def funB(odatis_key, odatis_number, aeris_key, aeris_number, model):
    print("B-" + str(aeris_number))

    return aeris_number, pipeline_multi_token_distance(select_token(odatis_key, odatis_number, lambda x: x.split("->")),
                                                       select_token(aeris_key, aeris_number, lambda x: x.split("->")),
                                                       model)


def get_val_from_path(path, dict_to_get, sep="."):
    x = dict_to_get
    for sub in path.split(sep):
        try:
            x = x[sub]
        except:
            return x
    return x


#


def get_max_path_length(model_df, delimiter):
    """
    Retourne la taille des chemins et le nombre d'occurence de cette taille dans le model
    :param model_df: list : list of path
    :return: int : max path size / number of node in the longest path in the model
    """
    aux_counter = 0
    for key_path in model_df:
        if len(key_path.split(delimiter)) > aux_counter:
            aux_counter = len(key_path.split(delimiter))
    return aux_counter


if __name__ == '__main__':
    path_to_mapping = "mappings/"
    path_to_models = "models/"
    path_to_res = "results/"
    print("Models have to be extracted in "+path_to_models+" folder.")
    print("Be careful : matching can take a lot of time !")
    # print("Don't forget to start mongodb docker container with port 27017 open on localhost.")
    # print("Starting model extraction from files in..." + path_to_models)
    # mongo_client = MongoClient("localhost:27017")
    # base_path = "."
    # model_dict = {}
    # model_examples_folder = list(os.walk(path_to_models))[0][1]

    print("Loading glove_vectors..")

    glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
    onlyfiles = [f for f in os.listdir(path_to_mapping) if
                 os.path.isfile(os.path.join(path_to_mapping, f)) and f.endswith(".csv")]

    res_files_list = [f for f in os.listdir(path_to_res) if
                      os.path.isfile(os.path.join(path_to_res, f)) and f.endswith(".txt")]

    print("Starting matching...")

    for ground_truth in onlyfiles:
        print(ground_truth)
        # res = {}
        parameters = ground_truth.split("_")
        modelA_name = parameters[0]
        modelB_name = parameters[2]
        delimiter = parameters[3]
        modelA = pd.read_csv(path_to_models + modelA_name + "_model.csv").set_index("Unnamed: 0").transpose()
        modelB = pd.read_csv(path_to_models + modelB_name + "_model.csv").set_index("Unnamed: 0").transpose()

        modelA = modelA.rename(columns=lambda x: x.replace(".", delimiter))
        modelB = modelB.rename(columns=lambda x: x.replace(".", delimiter))

        glove_dict_res = {}
        key_error = []
        word_error = []


        res_full={}
        for sub_path_A_size in range(get_max_path_length(modelA.columns, delimiter)):
            for sub_path_B_size in range(get_max_path_length(modelB.columns, delimiter)):
                print("Word2Vec matching ... ")

                if "Word2Vec_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
                     + str(sub_path_B_size) + "-sized_path.dict.txt" not in res_files_list:
                    res = {}
                    res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] = {}
                    start = time.time()
                    res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name]["results"] = \
                        pipeline_sum_euclid(list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
                                                                  lambda x: x.split(delimiter)))),
                                            list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
                                                                  lambda x: x.split(delimiter)))),
                                            glove_vectors)
                    end = time.time()
                    print("Time elapsed only for matching with euclidian distance for "+ modelA_name + " with "
                          + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
                          + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
                    res["euclidian_sum_pipeline/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] \
                        ["processing_time"] = str(end - start)
                    start = time.time()

                    res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] = {}

                    res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name]["results"] = \
                        pipeline_multi_token_distance(
                            list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
                                                  lambda x: x.split(delimiter)))),
                            list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
                                                  lambda x: x.split(delimiter)))),
                            glove_vectors)
                    end = time.time()
                    print("Time elapsed only for matching with word mover's distance for "+ modelA_name + " with "
                          + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
                          + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
                    res["multi_word/wikipedia_corpus_word2vec/" + modelA_name + "/" + modelB_name] \
                        ["processing_time"] = str(end - start)
                    with open(path_to_res+"Word2Vec_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"
                         + str(sub_path_B_size) + "-sized_path.dict.txt", "wb") as f:
                        pickle.dump(res, f)
                else:
                    print("Word2Vec_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
                     + str(sub_path_B_size) + "-sized_path.dict.txt skipped.")
                print("COMA matching ... ")
                if "COMA_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
                     + str(sub_path_B_size) + "-sized_path.dict.txt" not in res_files_list:
                    res = {}
                    print("Beggin matching for : " + "COMA"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
                     + str(sub_path_B_size) + "-sized_path.dict.txt" )
                    res["COMA/" + modelA_name + "/" + modelB_name] = {}
                    start = time.time()

                    res["COMA/" + modelA_name + "/" + modelB_name]["results"] = \
                        valentine_match(pd.DataFrame(columns=(list(set(select_token(modelA.columns, sub_path_A_size + 1, delimiter,
                                                                  lambda x: x.split(delimiter)))))),
                                            pd.DataFrame(columns=list(set(select_token(modelB.columns, sub_path_B_size + 1, delimiter,
                                                                  lambda x: x.split(delimiter))))), Coma(strategy="COMA_OPT"))
                    end = time.time()
                    print("Time elapsed only for matching with euclidian distance for "+ modelA_name + " with "
                          + str(sub_path_A_size +1) + " subpath size and " + modelB_name + " with "
                          + str(sub_path_B_size +1) + " subpath size and " +  " : " + str(end - start))
                    res["COMA/" + modelA_name + "/" + modelB_name] \
                        ["processing_time"] = str(end - start)
                    start = time.time()

                    with open(path_to_res+"COMA_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"
                         + str(sub_path_B_size) + "-sized_path.dict.txt", "wb") as f:
                        pickle.dump(res, f)
                else :
                    print("COMA_"+modelA_name+"_"+str(sub_path_A_size)+"-sizedpath_"+modelB_name+"_"\
                     + str(sub_path_B_size) + "-sized_path.dict.txt skipped.")