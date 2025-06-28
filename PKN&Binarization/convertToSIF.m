% =========================================================================
% FUNCTION TO CONVERT A TWO-COLUMN PKN INTO A THREE-COLUMN SIF FILE
% =========================================================================
%
% Description:
% This function reads a two-column, tab-separated file containing
% interactions (Regulator -> Target) and converts it into a three-column
% Simple Interaction Format (SIF) file. Since the interaction type is not
% specified in the source file, a generic type is used.
%
% The default generic interaction type is 'pd' (protein-protein interaction).
%
% Author: Your Name / AI Assistant
% Date: June 2025
%
function convertToSIF(inputFile, outputFile, interactionType)
    % Example Usage:
    % convertToSIF('pkn_hsa05235_symbols.txt', 'network.sif');

    % Set a default interaction type if one isn't provided
    if nargin < 3
        interactionType = 'pd';
    end

    fprintf('Starting conversion from TXT to SIF format...\n');
    fprintf('Input file: %s\n', inputFile);
    fprintf('Output file: %s\n', outputFile);
    fprintf('Using interaction type: ''%s''\n', interactionType);

    % Open the input and output files
    try
        inputFileID = fopen(inputFile, 'r');
        if inputFileID == -1
            error('Could not open input file: %s', inputFile);
        end
        
        outputFileID = fopen(outputFile, 'w');
        if outputFileID == -1
            fclose(inputFileID); % Close the input file before erroring
            error('Could not open output file for writing: %s', outputFile);
        end
    catch ME
        error('File operation failed. Error: %s', ME.message);
    end

    line_count = 0;
    % Read the input file line by line
    while ~feof(inputFileID)
        line = fgetl(inputFileID);
        
        % Skip empty lines
        if isempty(line)
            continue;
        end
        
        % Split the line by the tab character
        parts = strsplit(line, '\t');
        
        % Ensure the line has exactly two columns
        if numel(parts) == 2
            sourceNode = parts{1};
            targetNode = parts{2};
            
            % Write the new SIF-formatted line to the output file
            % Format: Source <space> InteractionType <space> Target
            fprintf(outputFileID, '%s %s %s\n', sourceNode, interactionType, targetNode);
            line_count = line_count + 1;
        end
    end

    % Close both files
    fclose(inputFileID);
    fclose(outputFileID);

    fprintf('------------------------------------------------------------\n');
    fprintf('PROCESS COMPLETE\n');
    fprintf('Successfully converted %d interactions.\n', line_count);
    fprintf('SIF file is ready: %s\n', outputFile);
    fprintf('------------------------------------------------------------\n');
end