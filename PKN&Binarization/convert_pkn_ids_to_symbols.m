% =========================================================================
% SCRIPT TO CONVERT A NUMERIC PKN (GENE IDS) TO A SYMBOL-BASED PKN
% =========================================================================
%
% Description:
% This script takes a two-column PKN file containing NCBI Gene IDs and
% an annotation file. It creates a new PKN file where the Gene IDs have
% been converted to their corresponding Gene Symbols.

%
% Author: Your Name / AI Assistant
% Date: June 2025
%
function convert_pkn_ids_to_symbols(numeric_pkn_file, annotation_file, output_symbol_pkn_file)
    % Example Usage:
    % convert_pkn_ids_to_symbols('pkn_hsa05235.txt', 'Human.GRCh38.p13.annot.tsv', 'pkn_hsa05235_symbols.txt');

    fprintf('Starting PKN ID to Symbol conversion...\n');

    % Step 1: Load the annotation file and create a lookup map
    fprintf('Step 1/4: Loading annotation file: %s\n', annotation_file);
    try
        opts_annot = detectImportOptions(annotation_file, 'FileType', 'text');
        opts_annot.Delimiter = '\t';
        % Ensure GeneID is read as a number and Symbol as a string
        opts_annot = setvartype(opts_annot, {'GeneID', 'Symbol'}, {'double', 'string'});
        annotationData = readtable(annotation_file, opts_annot);
    catch ME
        error('Failed to load annotation file. Error: %s', ME.message);
    end
    
    % Remove rows with missing symbols or IDs
    annotationData(ismissing(annotationData.Symbol) | isnan(annotationData.GeneID), :) = [];
    
    fprintf('Step 2/4: Building Gene ID -> Symbol map...\n');
    % Create a map for fast lookups. Convert numeric GeneID to string for the map key.
    idToSymbolMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
    for i = 1:height(annotationData)
        geneID_str = num2str(annotationData.GeneID(i));
        % Avoid overwriting existing entries to keep the first symbol found
        if ~isKey(idToSymbolMap, geneID_str)
            idToSymbolMap(geneID_str) = char(annotationData.Symbol(i));
        end
    end
    fprintf(' -> Map created with %d unique entries.\n', idToSymbolMap.Count);

    % Step 3: Read the numeric PKN file
    fprintf('Step 3/4: Reading numeric PKN file: %s\n', numeric_pkn_file);
    try
        numeric_pkn = readmatrix(numeric_pkn_file);
    catch ME
        error('Failed to read numeric PKN file. Ensure it is a two-column numeric file. Error: %s', ME.message);
    end

    % Step 4: Convert IDs and write to the new file
    fprintf('Step 4/4: Converting IDs and writing to: %s\n', output_symbol_pkn_file);
    fileID = fopen(output_symbol_pkn_file, 'w');
    if fileID == -1
        error('Could not open file for writing: %s', output_symbol_pkn_file);
    end
    
    converted_count = 0;
    skipped_count = 0;
    for i = 1:size(numeric_pkn, 1)
        regulatorID = num2str(numeric_pkn(i, 1));
        targetID    = num2str(numeric_pkn(i, 2));
        
        % Check if both IDs exist in our map
        if isKey(idToSymbolMap, regulatorID) && isKey(idToSymbolMap, targetID)
            regulatorSymbol = idToSymbolMap(regulatorID);
            targetSymbol    = idToSymbolMap(targetID);
            
            fprintf(fileID, '%s\t%s\n', regulatorSymbol, targetSymbol);
            converted_count = converted_count + 1;
        else
            skipped_count = skipped_count + 1;
        end
    end
    fclose(fileID);

    fprintf('------------------------------------------------------------\n');
    fprintf('PROCESS COMPLETE\n');
    fprintf('Successfully converted and wrote %d interactions.\n', converted_count);
    if skipped_count > 0
        fprintf('WARNING: Skipped %d interactions because one or both Gene IDs were not found in the annotation file.\n', skipped_count);
    end
    fprintf('Final symbol-based PKN is ready: %s\n', output_symbol_pkn_file);
    fprintf('------------------------------------------------------------\n');
end