function sim = CAGS(imageRef, imageDis)
% ========================================================================
% CAGS Index with automatic downsampling, Version 1.0£¬2020.1.6
% Copyright(c) 2019 Chenyang Shi£¬Yandan Lin
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------


labRef = RGB2Lab(imageRef);
labDis = RGB2Lab(imageDis);


[rows, cols, junk] = size(imageRef);

if ndims(imageRef) == 3 %images are colorful
    L1 = labRef(:,:,1);
    L2 = labDis(:,:,1);
    a1 = labRef(:,:,2);
    a2 = labDis(:,:,2);
    b1 = labRef(:,:,3);
    b2 = labDis(:,:,3);
end
%%%%%%%%%%%%%%%%%%%%%%%%%
% Downsample the image
%%%%%%%%%%%%%%%%%%%%%%%%%
minDimension = min(rows,cols);
F = max(1,round(minDimension / 256));
aveKernel = fspecial('average',F);

aveL1 = conv2(L1, aveKernel,'same');
aveL2 = conv2(L2, aveKernel,'same');
L1 = aveL1(1:F:rows,1:F:cols);
L2 = aveL2(1:F:rows,1:F:cols);

avea1 = conv2(a1, aveKernel,'same');
avea2 = conv2(a2, aveKernel,'same');
a1 = avea1(1:F:rows,1:F:cols);
a2 = avea2(1:F:rows,1:F:cols);

aveb1 = conv2(b1, aveKernel,'same');
aveb2 = conv2(b2, aveKernel,'same');
b1 = aveb1(1:F:rows,1:F:cols);
b2 = aveb2(1:F:rows,1:F:cols);
%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the CA indexes
%%%%%%%%%%%%%%%%%%%%%%%%%
Vab1 = (L1.^2 + a1.^2 + b1.^2).^(1/2);
Vab2 = (L2.^2 + a2.^2 + b2.^2).^(1/2);
Dab1 = ((100-L1).^2 + a1.^2 + b1.^2).^(1/2);
Dab2 = ((100-L2).^2 + a2.^2 + b2.^2).^(1/2);


%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the gradient map
%%%%%%%%%%%%%%%%%%%%%%%%%
dx = [3 0 -3; 10 0 -10;  3  0 -3]/16;
dy = [3 10 3; 0  0   0; -3 -10 -3]/16;
IxL1 = conv2(L1, dx, 'same');     
IyL1 = conv2(L1, dy, 'same');    
gradientMap1 = sqrt(IxL1.^2 + IyL1.^2);

IxL2 = conv2(L2, dx, 'same');     
IyL2 = conv2(L2, dy, 'same');    
gradientMap2 = sqrt(IxL2.^2 + IyL2.^2);

Kv = 0.02;%fixed
Kd = 0.02;%fixed
Kg = 50;  %fixed

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the CAGS index
%%%%%%%%%%%%%%%%%%%%%%%%%
VSimMatrix = (2 * Vab1 .* Vab2 + Kv) ./ (Vab1.^2 + Vab2.^2 + Kv);
DSimMatrix = (2 * Dab1 .* Dab2 + Kd) ./ (Dab1.^2 + Dab2.^2 + Kd);
gradientSimMatrix = (2*gradientMap1.*gradientMap2 + Kg) ./(gradientMap1.^2 + gradientMap2.^2 + Kg);
weight = max(Vab1, Vab2);

alpha = 0.1;%fixed


SimMatrix = gradientSimMatrix.* (VSimMatrix .^ alpha) .* DSimMatrix .* weight;
sim = sum(sum(SimMatrix)) / sum(weight(:));

return;

function labImage = RGB2Lab(image)

image = double(image);
normalizedR = image(:,:,1) / 255;
normalizedG = image(:,:,2) / 255;
normalizedB = image(:,:,3) / 255;

RSmallerOrEqualto4045 = normalizedR <= 0.04045;
RGreaterThan4045 = 1 - RSmallerOrEqualto4045;
tmpR = (normalizedR / 12.92) .* RSmallerOrEqualto4045;
tmpR = tmpR + power((normalizedR + 0.055)/1.055,2.4) .* RGreaterThan4045;

GSmallerOrEqualto4045 = normalizedG <= 0.04045;
GGreaterThan4045 = 1 - GSmallerOrEqualto4045;
tmpG = (normalizedG / 12.92) .* GSmallerOrEqualto4045;
tmpG = tmpG + power((normalizedG + 0.055)/1.055,2.4) .* GGreaterThan4045;

BSmallerOrEqualto4045 = normalizedB <= 0.04045;
BGreaterThan4045 = 1 - BSmallerOrEqualto4045;
tmpB = (normalizedB / 12.92) .* BSmallerOrEqualto4045;
tmpB = tmpB + power((normalizedB + 0.055)/1.055,2.4) .* BGreaterThan4045;

X = tmpR*0.4124564 + tmpG*0.3575761 + tmpB*0.1804375;
Y = tmpR*0.2126729 + tmpG*0.7151522 + tmpB*0.0721750;
Z = tmpR*0.0193339 + tmpG*0.1191920 + tmpB*0.9503041;

epsilon = 0.008856;	%actual CIE standard
kappa   = 903.3;	%actual CIE standard
 
Xr = 0.9642;	%reference white D50
Yr = 1.0;		%reference white
Zr = 0.8251;	%reference white

xr = X/Xr;
yr = Y/Yr;
zr = Z/Zr;

xrGreaterThanEpsilon = xr > epsilon;
xrSmallerOrEqualtoEpsilon = 1 - xrGreaterThanEpsilon;
fx = power(xr, 1.0/3.0) .* xrGreaterThanEpsilon;
fx = fx + (kappa*xr + 16.0)/116.0 .* xrSmallerOrEqualtoEpsilon;

yrGreaterThanEpsilon = yr > epsilon;
yrSmallerOrEqualtoEpsilon = 1 - yrGreaterThanEpsilon;
fy = power(yr, 1.0/3.0) .* yrGreaterThanEpsilon;
fy = fy + (kappa*yr + 16.0)/116.0 .* yrSmallerOrEqualtoEpsilon;

zrGreaterThanEpsilon = zr > epsilon;
zrSmallerOrEqualtoEpsilon = 1 - zrGreaterThanEpsilon;
fz = power(zr, 1.0/3.0) .* zrGreaterThanEpsilon;
fz = fz + (kappa*zr + 16.0)/116.0 .* zrSmallerOrEqualtoEpsilon;

[rows,cols,junk] = size(image);
labImage = zeros(rows,cols,3);
labImage(:,:,1) = 116.0 * fy - 16.0;
labImage(:,:,2) = 500.0 * (fx - fy);
labImage(:,:,3) = 200.0 * (fy - fz);
return;



