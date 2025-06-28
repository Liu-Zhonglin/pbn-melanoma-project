% Filename: parse_pbn_json.m
%
% PURPOSE:
% Helper function to parse a PBN JSON file into a MATLAB containers.Map.
% A map is like a Python dictionary, allowing easy access to nodes by name.
% This function must be in the same directory as the main analysis script.

function pbn_map = parse_pbn_json(filename)
    
    if ~exist(filename, 'file')
        error('File not found: %s. Make sure the JSON file path is correct.', filename);
    end
    
    try
        text = fileread(filename);
        data = jsondecode(text);
        pbn_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
        
        node_names = fieldnames(data.nodes);
        for i = 1:length(node_names)
            node = node_names{i};
            node_data = data.nodes.(node);
            
            % This handles the case where a node has no functions (e.g., an input node)
            if ~isfield(node_data, 'functions')
                node_data.functions = [];
            end
            
            pbn_map(node) = node_data;
        end
    catch ME
        error('Failed to parse the JSON file "%s". Check for syntax errors in the JSON. MATLAB error: %s', filename, ME.message);
    end
end