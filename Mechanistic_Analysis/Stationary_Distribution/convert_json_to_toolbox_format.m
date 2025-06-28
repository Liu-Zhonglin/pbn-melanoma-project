function model = convert_json_to_toolbox_format(json_model)
% Converts a PBN model from the JSON format to the specific matrix format
% required by the MATLAB PBN toolbox (F, varF, nf, nv, cij).
% VERSION: v6 - Final, "Paranoid" version. Uses a local helper function 
%          to sanitize every value extracted from the JSON structure.

    gene_list = fieldnames(json_model.nodes);
    n = length(gene_list);
    gene_to_idx = containers.Map(gene_list, 1:n);

    % --- Pass 1: Determine the dimensions (nf, nv) ---
    fprintf('  Pass 1: Determining model dimensions...\n');
    nf = zeros(1, n);
    nv_list = []; 

    for i = 1:n
        gene = gene_list{i};
        gene_data = json_model.nodes.(gene);
        
        functions_data = get_cell_array_safely(gene_data, 'functions');
        
        func_count_for_gene = 0;
        for j = 1:length(functions_data)
            func_data = functions_data{j};
            
            prob_val = get_numeric_safely(func_data, 'probability');
            
            if prob_val > 0
                func_count_for_gene = func_count_for_gene + 1;
                inputs_val = get_cell_array_safely(func_data, 'inputs');
                nv_list(end+1) = length(inputs_val);
            end
        end
        nf(i) = func_count_for_gene;
    end
    
    model.nf = nf;
    model.nv = nv_list;
    
    % --- Pass 2: Populate the matrices (F, varF, cij) ---
    fprintf('  Pass 2: Populating toolbox matrices...\n');
    max_nv = 0; if ~isempty(model.nv), max_nv = max(model.nv); end
    sum_nf = sum(model.nf);
    max_nf = 0; if ~isempty(model.nf), max_nf = max(model.nf); end

    F = -ones(2^max_nv, sum_nf);
    varF = -ones(max_nv, sum_nf);
    cij = -ones(max_nf, n);
    
    func_col_idx = 0;

    for i = 1:n
        gene = gene_list{i};
        gene_data = json_model.nodes.(gene);
        
        functions_data = get_cell_array_safely(gene_data, 'functions');
        
        prob_list = [];
        for j = 1:length(functions_data)
            func_data = functions_data{j};
            
            prob_val = get_numeric_safely(func_data, 'probability');
            
            if prob_val > 0
                func_col_idx = func_col_idx + 1;
                prob_list(end+1) = prob_val;
                
                truth_table = convert_to_truth_table_robust(func_data);
                
                inputs_val = get_cell_array_safely(func_data, 'inputs');
                num_inputs = length(inputs_val);
                
                for k = 1:num_inputs
                    varF(k, func_col_idx) = gene_to_idx(inputs_val{k});
                end
                
                F(1:2^num_inputs, func_col_idx) = truth_table;
            end
        end
        
        if ~isempty(prob_list)
            cij(1:length(prob_list), i) = prob_list / sum(prob_list);
        end
    end
    
    model.F = F;
    model.varF = varF;
    model.cij = cij;
    model.gene_list = gene_list;
end

% --- LOCAL HELPER FUNCTIONS ---

function num_val = get_numeric_safely(struct_data, field_name)
    % Defensively extracts a numeric scalar from a struct field.
    num_val = 0;
    if isfield(struct_data, field_name)
        temp_val = struct_data.(field_name);
        if iscell(temp_val), temp_val = temp_val{1}; end
        if ischar(temp_val), temp_val = str2double(temp_val); end
        if isnumeric(temp_val) && isscalar(temp_val)
            num_val = temp_val;
        end
    end
end

function cell_val = get_cell_array_safely(struct_data, field_name)
    % Defensively extracts a cell array from a struct field.
    cell_val = {};
    if isfield(struct_data, field_name)
        temp_val = struct_data.(field_name);
        if iscell(temp_val)
            cell_val = temp_val;
        elseif isstruct(temp_val) % Handles case of single object vs array
            cell_val = {temp_val};
        end
    end
end

function truth_table = convert_to_truth_table_robust(func_data)
    % Robustly generates a truth table from a function struct.
    func_str = '';
    if isfield(func_data, 'function'), func_str = func_data.function; end
    
    func_str = strrep(func_str, '&', '&&');
    func_str = strrep(func_str, '|', '||');
    func_str = strrep(func_str, '~', '~');

    if strcmp(func_str, '0'), truth_table = [0]; return;
    elseif strcmp(func_str, '1'), truth_table = [1]; return; end
    
    inputs = get_cell_array_safely(func_data, 'inputs');
    num_inputs = length(inputs);
    
    if num_inputs > 0
        truth_table = zeros(1, 2^num_inputs);
        for i = 0:(2^num_inputs - 1)
            input_vals = de2bi(i, num_inputs, 'left-msb');
            temp_expr = func_str;
            for k = 1:length(inputs)
                temp_expr = regexprep(temp_expr, ['\<' inputs{k} '\>'], num2str(input_vals(k)));
            end
            truth_table(i + 1) = eval(temp_expr);
        end
    else
        truth_table = eval(func_str);
    end
end