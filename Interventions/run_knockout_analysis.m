% =========================================================================
% SCRIPT: run_knockout_analysis.m
%
% PURPOSE:
%   Performs in silico knockout experiments on the non-responder PBN model
%   to identify "master switch" regulators of the resistant state.
%
% v4 REVISIONS:
%   - Expanded the list of knockout targets to include NRAS, MAPK3, and
%     PIK3CB to create a more comprehensive and persuasive analysis.
%
% WORKFLOW:
%   1. Loads the non-responder PBN model.
%   2. Starts a parallel pool of MATLAB workers.
%   3. Runs SSD simulations for the wild-type and all knockout models in parallel.
%   4. Generates a final report and plot comparing the results to assess
%      their therapeutic potential.
% =========================================================================

clc;
clear;
close all;

fprintf('--- Starting In Silico Knockout Analysis (Parallel Version, v4) ---\n\n');

%% --- 1. CONFIGURATION ---

% --- Path to the PBN Toolbox ---
TOOLBOX_PATH = '/Users/liuzhonglin/Desktop/PBN_Melanoma_Project/Improved/Toolbox/pbn-matlab-toolbox';
addpath(genpath(TOOLBOX_PATH));
fprintf('PBN toolbox added to path.\n\n');

% --- Files ---
NON_RESPONDER_JSON_FILE = 'non_responder_PBN_model_mi_loose.json';
OUTPUT_REPORT_FILE = 'knockout_analysis_report.txt';
OUTPUT_PLOT_FILE = 'knockout_comparison_plot.png';

% --- Genes to Knock Out ---
% Expanded list to create a more persuasive and comprehensive analysis.
genes_to_knockout = {'JUN', 'LOXL2', 'RELA', 'NRAS', 'MAPK3', 'PIK3CB'};

% --- Simulation Parameters ---
N_STEPS = 2000000;
PERTURBATION_PROB = 0.001;

% --- Phenotype Definitions ---
phenotypes = struct();
phenotypes.Responder_Phenotype.definition = {'AXL', 0; 'WNT5A', 0; 'ROR2', 1};
phenotypes.Dominant_Non_Responder.definition = {'AXL', 0; 'WNT5A', 0; 'ROR2', 0};
phenotypes.AXL_Driven_Non_Responder.definition = {'AXL', 1; 'WNT5A', 0};

%% --- 2. LOAD AND PREPARE BASE MODEL ---

fprintf('--> Loading and preparing the non-responder model...\n');
try
    json_data_nonres = jsondecode(fileread(NON_RESPONDER_JSON_FILE));
    model_nonres_wt = convert_json_to_toolbox_format(json_data_nonres);
    gene_list = model_nonres_wt.gene_list;
    num_genes = length(gene_list);
    fprintf('    Model loaded successfully.\n\n');
catch ME
    error('Failed to process the non-responder model. Error: %s', ME.message);
end

%% --- 3. RUN SIMULATIONS IN PARALLEL ---

% Start a parallel pool if one is not already running
if isempty(gcp('nocreate'))
    parpool;
end

% Define all simulation cases to run
sim_cases = ['WildType', genes_to_knockout];
num_sims = length(sim_cases);
pmf_results = cell(1, num_sims);

fprintf('--> Starting parallel simulations for %d cases (this may take a while)...\n', num_sims);

parfor i = 1:num_sims
    current_case = sim_cases{i};
    fprintf('    ...Starting simulation for: %s\n', current_case);
    
    % Create a local copy of the model for this worker
    local_model = model_nonres_wt; 
    
    if ~strcmp(current_case, 'WildType')
        % This is a knockout case, so modify the local model
        gene_to_ko = current_case;
        gene_idx_to_ko = find(strcmp(gene_list, gene_to_ko));

        if ~isempty(gene_idx_to_ko)
            % Modify the PBN model to clamp the gene to 0
            start_col = sum(local_model.nf(1:gene_idx_to_ko-1)) + 1;
            end_col = start_col + local_model.nf(gene_idx_to_ko) - 1;
            local_model.F(:, start_col:end_col) = 0;
            local_model.cij(:, gene_idx_to_ko) = 0;
            local_model.cij(1, gene_idx_to_ko) = 1;
        end
    end
    
    % Run simulation on the local model (wild-type or knockout)
    pmf_results{i} = run_simulation(local_model, N_STEPS, PERTURBATION_PROB);
    fprintf('    ...Finished simulation for: %s\n', current_case);
end

fprintf('--> All parallel simulations complete.\n\n');

% Consolidate results from the cell array into a Map for easier reporting
results = containers.Map('KeyType', 'char', 'ValueType', 'any');
sim_cases_out = cell(1, num_sims);
for i = 1:num_sims
    case_name = sim_cases{i};
    if ~strcmp(case_name, 'WildType')
        case_name = [case_name, '_KO'];
    end
    results(case_name) = pmf_results{i};
    sim_cases_out{i} = case_name;
end


%% --- 4. GENERATE FINAL REPORT AND PLOT ---

fprintf('--> Generating final reports and plot...\n');

% Populate gene indices in phenotype definitions
phenotype_names = fieldnames(phenotypes);
for i = 1:length(phenotype_names)
    p_name = phenotype_names{i};
    def = phenotypes.(p_name).definition;
    for j = 1:size(def, 1)
        phenotypes.(p_name).definition{j, 3} = find(strcmp(gene_list, def{j, 1}));
    end
end

% --- Generate Text Report ---
fid = fopen(OUTPUT_REPORT_FILE, 'w');
fprintf(fid, '--- In Silico Knockout Analysis Report ---\n\n');
fprintf(fid, 'Comparison of phenotype probabilities in the Wild-Type (WT) Non-Responder model versus single-gene Knockout (KO) models.\n\n');

header_str = sprintf('%-30s', 'Phenotype');
for i = 1:length(sim_cases_out)
    header_str = [header_str, sprintf(' | %-15s', sim_cases_out{i})];
end
fprintf(fid, '%s\n', header_str);
fprintf(fid, '%s\n', repmat('-', 1, length(header_str)+2*length(sim_cases_out)) );

phenotype_prob_matrix = zeros(length(phenotype_names), length(sim_cases_out));

for i = 1:length(phenotype_names)
    p_name = phenotype_names{i};
    p_name_display = strrep(p_name, '_', ' ');
    row_str = sprintf('%-30s', p_name_display);
    
    for j = 1:length(sim_cases_out)
        pmf = results(sim_cases_out{j});
        prob = calculate_phenotype_probability(pmf, phenotypes.(p_name).definition, num_genes);
        phenotype_prob_matrix(i, j) = prob;
        row_str = [row_str, sprintf(' | %-15.4f', prob)];
    end
    fprintf(fid, '%s\n', row_str);
end
fclose(fid);
fprintf('Knockout analysis report saved to: %s\n', OUTPUT_REPORT_FILE);


% --- Generate and Save Bar Plot ---
fig = figure('Visible', 'off');

% --- Corrected Plotting Logic ---
% Ensure 'WildType' is the first bar for comparison
wt_idx = find(strcmp(sim_cases_out, 'WildType'));
other_idx = find(~strcmp(sim_cases_out, 'WildType'));
plot_order = [wt_idx, other_idx];
plot_labels = sim_cases_out(plot_order);
plot_data = phenotype_prob_matrix(:, plot_order)';

b = bar(plot_data, 'grouped');
% --- End Corrected Logic ---

set(gca, 'XTickLabel', plot_labels);
xtickangle(45);
ylabel('Total Phenotype Probability');
title('Effect of In Silico Gene Knockouts on Non-Responder Phenotypes');
legend(strrep(phenotype_names, '_', ' '), 'Location', 'northwest');
grid on;
ylim([0 1]);
set(gcf, 'Position', [100, 100, 1200, 700]); % Increased width for more bars

saveas(fig, OUTPUT_PLOT_FILE);
fprintf('Knockout comparison plot saved to: %s\n', OUTPUT_PLOT_FILE);

fprintf('\n====== ANALYSIS COMPLETE ======\n');


%% --- HELPER FUNCTIONS ---

function pmf = run_simulation(model, n_steps, p)
    % Stripped-down simulation function for parallel execution.
    n = length(model.nf);
    x = rand(1, n) > 0.5;
    pmf = zeros(1, 2^n);
    pow_vec = 2.^[n-1:-1:0]';
    for k = 1:n_steps
        x = pbnNextState(x, model.F, model.varF, model.nf, model.nv, model.cij, p);
        decx = x * pow_vec + 1;
        pmf(decx) = pmf(decx) + 1;
    end
    pmf = pmf / n_steps;
end

function total_prob = calculate_phenotype_probability(pmf, phenotype_def, num_genes)
    total_prob = 0;
    for i = 0:(2^num_genes - 1)
        bin_vector = de2bi(i, num_genes, 'left-msb');
        is_match = true;
        for j = 1:size(phenotype_def, 1)
            gene_idx = phenotype_def{j, 3};
            required_state = phenotype_def{j, 2};
            if bin_vector(gene_idx) ~= required_state
                is_match = false;
                break;
            end
        end
        if is_match
            total_prob = total_prob + pmf(i + 1);
        end
    end
end
