from bs4 import BeautifulSoup

def generate_reader_lexical(test_file_fp):
    f = open(test_file_fp,'r', encoding='utf8')
    data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    lexelt_list = bs_data.find_all("lexelt")
    #print(lexelt_list[0], len(lexelt_list))
    reader_lexical = {}
    for i_mainword in range(len(lexelt_list)):
        main_word = lexelt_list[i_mainword]["item"]
        reader_lexical[main_word] = []
        instance_list = lexelt_list[i_mainword].find_all("instance")
        for j_instance in range(len(instance_list)):
            instance_id, clean_context, target_id = extract_instance_relevant_informations(instance_list[j_instance])
            reader_lexical[main_word].append({"instance_id":instance_id, "clean_context":clean_context, "target_id":target_id})
    return reader_lexical

def extract_instance_relevant_informations(instance):
    instance_id = int(instance["id"])
    context = instance.find("context")
    split_context = str(context).split(" ")
    target_index = -1
    for i in range(len(split_context)):
        if "<head>" in split_context[i]:
            target_index = i
    assert target_index != -1
    clean_context = context.text
    target_word = clean_context.split(" ")[target_index]
    
    return instance_id, clean_context, target_index

#print(generate_reader_lexical("dataset/test/lexsubfr_semdis2014_test.xml"))