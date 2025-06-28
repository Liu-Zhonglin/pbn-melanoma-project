% =========================================================================
% SCRIPT: run_ssd_analysis_matlab_v5.m
%
% PURPOSE:
%   Calculates and compares the steady-state distribution (SSD) for the
%   responder and non-responder PBN models. It then calculates the
%   marginal probability (steady-state expression level) for each gene.
%
% v5 REVISIONS:
%   - ADDED: A new analysis section to calculate the steady-state expression
%     level for each individual gene.
%   - ADDED: Saves the gene expression results to 'gene_expression_report.txt'.
%   - ADDED: Generates and saves a bar plot comparing gene expression
%     levels between the two models ('gene_expression_comparison.png').
%
% WORKFLOW:
%   1. Adds the PBN toolbox to the MATLAB path.
%   2. Loads the .json models and prepares them for simulation.
%   3. Runs a long simulation to compute the SSD.
%   4. Calculates the total probability for predefined phenotype regions.
%   5. Calculates the steady-state expression level for each gene.
%   6. Generates and saves all reports and plots.
% =========================================================================

clc;
clear;
close all;

fprintf('--- Starting MATLAB-based PBN Analysis (v5) ---\n\n');

%% --- 1. CONFIGURATION ---

% --- Paths ---
TOOLBOX_PATH = '/Users/liuzhonglin/Desktop/PBN_Melanoma_Project/Improved/Toolbox/pbn-matlab-toolbox';
addpath(genpath(TOOLBOX_PATH));
fprintf('PBN toolbox added to path.\n\n');

% --- Model and Output Files ---
RESPONDER_JSON_FILE = 'responder_PBN_model_mi_loose.json';
NON_RESPONDER_JSON_FILE = 'non_responder_PBN_model_mi_loose.json';
PHENOTYPE_REPORT_FILE = 'phenotype_ssd_report.txt';
GENE_EXPRESSION_REPORT_FILE = 'gene_expression_report.txt';
GENE_EXPRESSION_PLOT_FILE = 'gene_expression_comparison.png';

% --- Simulation Parameters ---
N_STEPS = 2000000;
PERTURBATION_PROB = 0.001;

% --- Phenotype Definitions ---
phenotypes = struct();
phenotypes.Dominant_Non_Responder.definition = {'AXL', 0; 'WNT5A', 0; 'ROR2', 0};
phenotypes.Responder_Phenotype.definition    = {'AXL', 0; 'WNT5A', 0; 'ROR2', 1};
phenotypes.WNT_Driven_Non_Responder.definition = {'WNT5A', 1; 'ROR2', 1};
phenotypes.AXL_Driven_Non_Responder.definition = {'AXL', 1; 'WNT5A', 0};

%% --- 2. PROCESS MODELS AND RUN SIMULATIONS ---

results = struct();

fprintf('--> Processing Responder Model...\n');
try
    json_data_res = jsondecode(fileread(RESPONDER_JSON_FILE));
    model_res = convert_json_to_toolbox_format(json_data_res);
    fprintf('    Simulating for %d steps...\n', N_STEPS);
    pmf_res = run_simulation_with_progress(model_res, N_STEPS, PERTURBATION_PROB);
    results.responder.pmf = pmf_res;
    results.responder.gene_list = model_res.gene_list;
    fprintf('    Responder simulation complete.\n\n');
catch ME
    error('Failed to process responder model. Error: %s', ME.message);
end

fprintf('--> Processing Non-Responder Model...\n');
try
    json_data_nonres = jsondecode(fileread(NON_RESPONDER_JSON_FILE));
    model_nonres = convert_json_to_toolbox_format(json_data_nonres);
    fprintf('    Simulating for %d steps...\n', N_STEPS);
    pmf_nonres = run_simulation_with_progress(model_nonres, N_STEPS, PERTURBATION_PROB);
    results.non_responder.pmf = pmf_nonres;
    results.non_responder.gene_list = model_nonres.gene_list;
    fprintf('    Non-Responder simulation complete.\n\n');
catch ME
    error('Failed to process non-responder model. Error: %s', ME.message);
end

%% --- 3. PHENOTYPE PROBABILITY ANALYSIS ---

gene_list = results.responder.gene_list;
num_genes = length(gene_list);
phenotype_names = fieldnames(phenotypes);

for i = 1:length(phenotype_names)
    p_name = phenotype_names{i};
    def = phenotypes.(p_name).definition;
    for j = 1:size(def, 1)
        phenotypes.(p_name).definition{j, 3} = find(strcmp(gene_list, def{j, 1}));
    end
end

fid = fopen(PHENOTYPE_REPORT_FILE, 'w');
fprintf(fid, '--- Phenotype Steady-State Distribution Results ---\n\n');
fprintf(fid, '%-30s | %-20s | %-20s\n', 'Phenotype', 'Responder Prob.', 'Non-Responder Prob.');
fprintf(fid, '%s\n', repmat('-', 1, 75));
for i = 1:length(phenotype_names)
    p_name = phenotype_names{i};
    prob_res = calculate_phenotype_probability(results.responder.pmf, phenotypes.(p_name).definition, num_genes);
    prob_nonres = calculate_phenotype_probability(results.non_responder.pmf, phenotypes.(p_name).definition, num_genes);
    fprintf(fid, '%-30s | %-20.6f | %-20.6f\n', strrep(p_name, '_', ' '), prob_res, prob_nonres);
end
fclose(fid);
fprintf('Phenotype probability report saved to: %s\n', PHENOTYPE_REPORT_FILE);


%% --- 4. GENE EXPRESSION LEVEL ANALYSIS ---

fprintf('\n--> Calculating steady-state gene expression levels...\n');

% Calculate expression levels for each model
res_expr = calculate_gene_expression(results.responder.pmf, num_genes);
non_res_expr = calculate_gene_expression(results.non_responder.pmf, num_genes);

% --- Save Gene Expression Report ---
fid = fopen(GENE_EXPRESSION_REPORT_FILE, 'w');
fprintf(fid, '--- Steady-State Gene Expression Levels ---\n\n');
fprintf(fid, '%-10s | %-25s | %-25s\n', 'Gene', 'Responder Expression Prob.', 'Non-Responder Expression Prob.');
fprintf(fid, '%s\n', repmat('-', 1, 70));
for i = 1:num_genes
    fprintf(fid, '%-10s | %-25.6f | %-25.6f\n', gene_list{i}, res_expr(i), non_res_expr(i));
end
fclose(fid);
fprintf('Gene expression report saved to: %s\n', GENE_EXPRESSION_REPORT_FILE);

% --- Generate and Save Bar Plot ---
fig = figure('Visible', 'off'); % Create figure but keep it hidden
bar_data = [res_expr', non_res_expr'];
b = bar(bar_data);
set(gca, 'XTickLabel', gene_list);
xtickangle(45);
ylabel('Steady-State Expression Probability');
title('Comparison of Steady-State Gene Expression');
legend({'Responder', 'Non-Responder'}, 'Location', 'northwest');
grid on;
set(gcf, 'Position', [100, 100, 900, 600]); % Make figure larger for readability

saveas(fig, GENE_EXPRESSION_PLOT_FILE);
fprintf('Gene expression plot saved to: %s\n', GENE_EXPRESSION_PLOT_FILE);

fprintf('\n====== ANALYSIS COMPLETE ======\n');


%% --- HELPER FUNCTIONS ---

function pmf = run_simulation_with_progress(model, n_steps, p)
    n = length(model.nf);
    x = rand(1, n) > 0.5;
    pmf = zeros(1, 2^n);
    pow_vec = 2.^[n-1:-1:0]';
    progress_msg_len = 0;
    for k = 1:n_steps
        x = pbnNextState(x, model.F, model.varF, model.nf, model.nv, model.cij, p);
        decx = x * pow_vec + 1;
        pmf(decx) = pmf(decx) + 1;
        if mod(k, n_steps/100) == 0
            progress = (k / n_steps) * 100;
            msg = sprintf('      ...Simulation progress: %3.0f%%', progress);
            fprintf([repmat('\b', 1, progress_msg_len), msg]);
            progress_msg_len = numel(msg);
        end
    end
    fprintf('\n');
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

function gene_expr = calculate_gene_expression(pmf, num_genes)
    % Calculates the marginal probability of each gene being ON.
    gene_expr = zeros(1, num_genes);
    for i = 0:(2^num_genes - 1)
        bin_vector = de2bi(i, num_genes, 'left-msb');
        % Find which genes are ON in this state
        on_genes = find(bin_vector == 1);
        % Add the probability of this state to the total for each ON gene
        if ~isempty(on_genes)
            gene_expr(on_genes) = gene_expr(on_genes) + pmf(i + 1);
        end
    end
end
