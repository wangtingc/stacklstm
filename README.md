================
project structure:
    * data:
        - ptb // for ppl 
        - stb // for clf
    * models:
        - stackLSTM
    * utils:
        - batchIterator
        - data_loader
        - ptb_loader
        - misc
    * examples:
        - ptb
        - stb


================
Preprocess: PtLoader
    @ func:
        - __init__          : init raw data path and data directory
        - load_data         : load the entire data
        - _load_btrees      : if no exisiting data, load binarized trees
        - _load_tree        : if no exisiting btrees, load parsed trees
        - _init_parser      : init a parser for parsing the raw data
        - _parse            : parse first for better accuracy
        - _build_vocab      : build the vocab from trees
        - _replace_by_widx  : replace the leaves with word index
        - _convert_to_binary: special case - if there is only one word in the sentence, parser won't return a Tree but a string
        - tree_to_btree     : convert constituent tree to binarized tree
        - proc: convert     : trees to list, including [x, a, p]
        - _proc_single      : proc single sample to [xi, ai, pi]


===============
Preprocess: SnilLoader
    @ func:
        - __init__          : init raw data path and data direcotry
        - load_data         : load the entire data
        - _read_json_data   : read raw data (json format)
        - _build_vocab      : build vocab from parsed tree in raw data
        - _json_to_data     : convert json format data to the data used for stack lstm
        - _proc_single      : convert single json format sample


===============
HelperLayers: StackLSTM
    @ member:
        - W_[l|r|e]_to_[l|r|e|o|c]_s
        - b_[l|r|e|c|o]_s
        - W_[i|h]_to_[i|f|o|c]_t
        - b_[i|f|o|c]_t
    @ func:
        - track             : one step for track lstm, same as LSTM
        - composite         : composite function for stack lstm, need l, r and e
        - step              : one step for stack lstm
        - get_output_for    : interface for stack lstm
