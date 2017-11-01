import re
import matplotlib.pyplot as plt

def extract_lines(filename,filter_str=None,filter_func=None):
    "extract lines using `filter_str` or `filter_func`"
    assert (filter_str is not None) or (filter_func is not None)
    if (filter_func is None):
        filter_func = lambda x:x.find(filter_str) !=-1
    lines = open(filename).readlines()
    f_lines = []
    for line in lines:
        if (filter_func(line)):
            f_lines.append(line)
    return f_lines

def _extract_var_values(var_names, var_indexes, var_types, lines):
    "return a dict: {var_name: [list of values]}"
    var_value_dict = {name: [] for name in var_names}
    for line in lines:
        items = re.findall(r"[-+]?\d*\.\d+|\d+",line)
        for i in range(len(var_names)):
            index = var_indexes[i]
            typecast = float if var_types[i] == "float" else int
            var_value_dict[var_names[i]].append(typecast(items[index]))
    return var_value_dict

def plot_trend_graph(var_names,var_indexes,var_types,var_colors,lines):
    """
    :param var_names:
    :param var_indexes: if the var to plot is the 1st number in a line, its `var_index` should be 0
    :param var_types: "float" or "int"
    :return:
    """
    assert len(var_names) == len(var_indexes) == len(var_types)
    var_color_dict = {var_names[i]:var_colors[i] for i in range(len(var_names))}
    # get values
    var_value_dict = _extract_var_values(var_names=var_names, var_types=var_types, var_indexes=var_indexes, lines=lines)
    # plot graph
    x = range(len(lines))
    for name,values in var_value_dict.items():
        plt.plot(x,values,color=var_color_dict[name])
    plt.show()