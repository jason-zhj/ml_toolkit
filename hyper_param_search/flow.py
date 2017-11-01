"""
This runs the learning procedure from training to testing
"""
import os

#TODO: 1. implement test func
#TODO: 2. implement the following 3 funcs
#TODO: 3. integrate this with hyper-parameter search

def _save_model(model_name,model_obj,save_path):
    "this saves the model with the name `model_name`.model"
    pass

def _save_record_files(filename, content, save_path):
    "this saves `content` to the given `filename`"
    pass

def _save_results_csv(train_results,test_results,save_path):
    pass

def run_flow(record_path,train_func,test_func,params,param_id):
    ## 1. run training
    train_output = train_func(params)
    models, train_results, train_records = train_output["models"],train_output["results"],train_output["records"]

    ## 2. save training outputs
    # save model, models should be a dict {model_name,model_obj}
    model_save_path = os.path.join(record_path,"models",param_id)
    for name,obj in models.items():
        _save_model(model_name=name,model_obj=obj,save_path=model_save_path)

    # save training records, `train_records` should be a dict {filename:content}
    train_record_save_path = os.path.join(record_path,"train_records",param_id)
    for filename,content in train_records:
        _save_record_files(filename=filename, content=content, save_path=train_record_save_path)

    ## 3. run testing
    test_output = test_func(params,models)
    test_results, test_records = test_output["results"],test_output["records"]

    # save test records, `test_records` should be a dict {filename:content}
    test_record_save_path = os.path.join(record_path, "test_records", param_id)
    for filename,content in train_records:
        _save_record_files(filename=filename, content=content, save_path=test_record_save_path)

    # save training and testing results
    result_save_path = os.path.join(record_path,"result.csv")
    _save_results_csv(train_results=train_results,test_results=test_results,save_path=result_save_path)

