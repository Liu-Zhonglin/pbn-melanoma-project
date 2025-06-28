function [F_single, varF_single, nv_single] = get_bn_from_pbn(model, choices)
% Extracts the F, varF, and nv matrices for a single deterministic BN
% defined by the 'choices' vector from the full PBN 'model'.

    n = length(model.nf);
    func_col_offsets = [0, cumsum(model.nf)]; % Find start column for each gene
    
    % Determine the absolute column index in the big F matrix for each choice
    chosen_cols = arrayfun(@(i) func_col_offsets(i) + choices(i), 1:n);
    
    nv_single = model.nv(chosen_cols);
    max_nv_single = max(nv_single);
    
    F_single = model.F(1:2^max_nv_single, chosen_cols);
    varF_single = model.varF(1:max_nv_single, chosen_cols);
end