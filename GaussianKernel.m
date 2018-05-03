function y = GaussianKernel(x,z,sigma)
if ~isvector(x)||~isvector(z)
    error('Input must be a vector')
end
if size(x)~=size(z)
    error('Input should be two vectors with the same size')
end
    y = exp((-1) * norm(x - z)^2/2/sigma^2);    
end
    