function sim = VCGS(image1, image2)
% ========================================================================
% VCGS Index with automatic downsampling, Version 1.0£¬2020.1.6
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
% 
% 
%----------------------------------------------------------------------
%
%Input : (1) image1: the first image being compared, which is a RGB image
%        (2) image2: the second image being compared, which is a RGB image
%
%Output: sim: the similarity score between two images, a real number
%        
%-----------------------------------------------------------------------
labRef = RGB2Lab(image1);
labDis = RGB2Lab(image2);

Kvs = 1.25;%fixed
Kc = 200;
Kg = 60;  %fixed
alpha = 0.4;%fixed
lambda = 0.02;%fixed
sigmaF = 1.34;%fixed donot change
omega0 = 0.0210;%fixed
sigmaC = 0.001;%fixed

%compute the visual saliency map with color appearance
[saliencyMap1] = VSca(labRef,sigmaF,omega0,sigmaC);
[saliencyMap2] = VSca(labDis,sigmaF,omega0,sigmaC);
[rows, cols, junk] = size(image1);

if ndims(image1) == 3 %images are colorful
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



aveSM1 = conv2(saliencyMap1, aveKernel,'same');
aveSM2 = conv2(saliencyMap2, aveKernel,'same');
saliencyMap1 = aveSM1(1:F:rows,1:F:cols);
saliencyMap2 = aveSM2(1:F:rows,1:F:cols);



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

gradientSimMatrix = (2*gradientMap1.*gradientMap2 + Kg) ./(gradientMap1.^2 + gradientMap2.^2 + Kg);
%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the VSCA
%%%%%%%%%%%%%%%%%%%%%%%%%

VSSimMatrix = (2 * saliencyMap1 .* saliencyMap2 + Kvs) ./ (saliencyMap1.^2 + saliencyMap2.^2 + Kvs);

ASimMatrix = (2 * a1 .* a2 + Kc) ./ (a1.^2 + a2.^2 + Kc);
BSimMatrix = (2 * b1 .* b2 + Kc) ./ (b1.^2 + b2.^2 + Kc);

weight = max(saliencyMap1, saliencyMap2);

SimMatrixC = (gradientSimMatrix .^ alpha) .* VSSimMatrix  .*real((ASimMatrix .* BSimMatrix) .^ lambda).* weight;

sim = sum(sum(SimMatrixC)) / sum(weight(:));


return;
%===================================
function [VSMap] = VSca(labimage,sigmaF,omega0,sigmaC)
% ========================================================================
%
%Input : image: an uint8 RGB image with dynamic range [0, 255] for each
%channel
%        
%Output: VSMap: the visual saliency map extracted by the SDSP algorithm.
%Data range for VSMap is [0, 255]. So, it can be regarded as a common
%gray-scale image.
%        
%-----------------------------------------------------------------------
%convert the image into LAB color space
[rows, cols, junk] = size(labimage);


LChannel = labimage(:,:,1);
AChannel = labimage(:,:,2);
BChannel = labimage(:,:,3);
Vab = (LChannel.^2 + AChannel.^2 + BChannel.^2).^(1/2);
Dab = ((100-LChannel).^2 + AChannel.^2 + BChannel.^2).^(1/2);

VFFT = fft2(double(Vab));
DFFT = fft2(double(Dab));

%color appearance with log-gabor filter have a bias towards attention
[rows, cols, junk] = size(labimage);
LG = logGabor(rows,cols,omega0,sigmaF);
FinalVResult = real(ifft2(VFFT.*LG));
FinalDResult = real(ifft2(DFFT.*LG));


SFMap = sqrt(FinalVResult.^2 + FinalDResult.^2 );

%color appearance have a bias towards attention


maxA = max(Vab(:));
minA = min(Vab(:));
normalizedA = (Vab - minA) / (maxA - minA);

maxB = max(Dab(:));
minB = min(Dab(:));
normalizedB = (Dab - minB) / (maxB - minB);

labDistSquare = normalizedA.^2 + normalizedB.^2;
SCMap = 1 - exp(-labDistSquare / (sigmaC^2));


VSMap = SFMap .* SCMap;
VSMap =  imresize(VSMap, [rows, cols],'bilinear');
VSMap = mat2gray(VSMap);



return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function LG = logGabor(rows,cols,omega0,sigmaF)
     [u1, u2] = meshgrid(([1:cols]-(fix(cols/2)+1))/(cols-mod(cols,2)), ...
			            ([1:rows]-(fix(rows/2)+1))/(rows-mod(rows,2)));
     mask = ones(rows, cols);
     for rowIndex = 1:rows
         for colIndex = 1:cols
             if u1(rowIndex, colIndex)^2 + u2(rowIndex, colIndex)^2 > 0.25
                 mask(rowIndex, colIndex) = 0;
             end
         end
     end
     u1 = u1 .* mask;
     u2 = u2 .* mask;
     
     u1 = ifftshift(u1);  
     u2 = ifftshift(u2);
     
     radius = sqrt(u1.^2 + u2.^2);    
     radius(1,1) = 1;
            
     LG = exp((-(log(radius/omega0)).^2) / (2 * (sigmaF^2)));  
     LG(1,1) = 0; 
return;