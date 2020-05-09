function closestQuadruple = MBVQIdentifier(R,G,B)
%MBVQIDENTIFIER Summary of this function goes here
%Take R(Red), G(Green), B(Blue) value and return corresponding the closest
%Quadruple.
%   Detailed explanation goes here
%Algorithm is from Discussion Note Week4 EE569
if((R+G)>255)
    if((G+B)>255)
        if((R+G+B)>510)
            closestQuadruple = 'CMYW';
        else
            closestQuadruple = 'MYGC';
        end
    else
        closestQuadruple = 'RGMY';
    end
else
    if(~((G+B)>255))
        if(~((R+G+B)>255))
            closestQuadruple = 'KRGB';
        else
            closestQuadruple = 'RGBM';
        end
    else
        closestQuadruple = 'CMGB';
    end
end

end

