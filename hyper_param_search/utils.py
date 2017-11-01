
class EmptyParamObject():
    pass

def create_obj_from_dict(dict_obj):
    param_obj = EmptyParamObject()
    for key in dict_obj.keys():
        setattr(param_obj,key,dict_obj[key])
    return param_obj



def write_record_to_csv(param_dict,result_dict,path,first_write,delimiter="\t"):
    combined_dict = {**param_dict,**result_dict}
    if (first_write):
        # write the titles
        open(path,"a").write(delimiter.join(combined_dict.keys()) + "\n")

    # write data
    open(path,"a").write(delimiter.join(combined_dict.values()) + "\n")