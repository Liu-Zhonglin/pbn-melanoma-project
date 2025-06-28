function prob = get_bn_probability(model, choices)
% Calculates the probability of a single BN based on the 'choices' vector.
    n = length(model.nf);
    prob_vec = zeros(1, n);
    for i = 1:n
        prob_vec(i) = model.cij(choices(i), i);
    end
    prob = prod(prob_vec);
end