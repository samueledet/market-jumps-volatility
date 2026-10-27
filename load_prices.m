function S = load_prices(file)
%LOAD_PRICES Read daily closing prices from a CSV file.
%   S = LOAD_PRICES(file) accepts either a single column of prices or a
%   file whose last column holds prices (for example a FRED download with
%   date and value columns). Missing values ("." or empty) are dropped.
T = readtable(file, 'TreatAsMissing', '.');
S = T{:, end};
if iscell(S)
    S = str2double(S);
end
S = S(~isnan(S) & S > 0);
end
