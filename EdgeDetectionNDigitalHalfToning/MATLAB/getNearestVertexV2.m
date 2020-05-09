function vertex = getNearestVertexV2(mbvq, R, G, B)
% getNearestVertex: find nearst vertex in the given MBVQ for a target pixel
% 
% INPUT:
% mbvq (char array): the mbvq the target pixel is related to
% R, G, B (range:0~1): the R, G, B channel value of the target pixel
% 
% OUPUT:
% vertex (array): array that represents one of 8 vertexes
% 
% Ref: "Color Diffusion: Error-Diffusion for Color Halftones
% by Shaked, Arad, Fitzhugh, and Sobel -- HP Labs
% Hewlett-Packard Laboratories TR 96-128
% and Electronic Imaging, Proc. SPIE 3648, 1999
% Adapted from Halftoning Toolbox Version 1.2 released July 2005 (Univ. of Texas)
%Corresponding Color vertex
% black: vertex = [0,0,0];
% Blue: vertex = [0,0,255];
% Green: vertex = [0,255,0];
% Red: vertex = [255,0,0];
% Cyan: vertex = [0,255,255];
% Magenta: vertex = [255,0,255];
% Yellow: vertex = [255,255,0];
% White: vertex = [255,255,255];




%Convert RGB values to values in [0,1]
R = R/255;
G = G/255;
B = B/255;

% No.1 for CMYW
    if (mbvq == 'CMYW')
        %vertex = 'white';
        vertex = [255,255,255];
        if (B < 0.5)
            if (B <= R)
                if (B <= G)                  
                    %vertex = 'yellow';
                     vertex = [255,255,0];
                end
            end
        end
        if (G < 0.5)
            if (G <= B)
                if (G <= R)
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                end
            end
        end
        if (R < 0.5)
            if (R <= B)
                if (R <= G)
                    %vertex = 'cyan';
                    vertex = [0,255,255];
                end
            end
        end
    end


% No.2 for MYGC
    if (mbvq == 'MYGC')
        %vertex = 'magenta'; 
        vertex = [255,0,255];
        if (G >= B)
            if (R >= B)
                if (R >= 0.5)
                    %vertex = 'yellow';
                    vertex = [255,255,0];
                else
                    %vertex = 'green';
                    vertex = [0,255,0];
                end
            end
        end
        if (G >= R)
            if (B >= R)
                if (B >= 0.5)
                    %vertex = 'cyan';
                    vertex = [0,255,255];
                else
                    %vertex = 'green';
                    vertex = [0,255,0];
                end
            end
        end
    end


% No.3 for RGMY
    if (mbvq == 'RGMY')
        if (B > 0.5)
            if (R > 0.5)
                if (B >= G)
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                else
                    %vertex = 'yellow';
                    vertex = [255,255,0];
                end
            else
                if (G > B + R)
                    %vertex = 'green';
                    vertex = [0,255,0];
                else 
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                end
            end
        else
            if (R >= 0.5)
                if (G >= 0.5)
                    %vertex = 'yellow';
                    vertex = [255,255,0];
                else
                    %vertex = 'red';
                    vertex = [255,0,0];
                end
            else
                if (R >= G)
                    %vertex = 'red';
                    vertex = [255,0,0];
                else
                    %vertex = 'green';
                    vertex = [0,255,0];
                end
            end
        end
    end


% No.4 for KRGB
    if (mbvq == 'KRGB')
        %vertex = 'black';
        vertex = [0,0,0];
        if (B > 0.5)
            if (B >= R)
                if (B >= G)
                    %vertex = 'blue';
                    vertex = [0,0,255];
                end
            end
        end
        if (G > 0.5)
            if (G >= B)
                if (G >= R)
                    %vertex = 'green';
                    vertex = [0,255,0];
                end
            end
        end
        if (R > 0.5)
            if (R >= B)
                if (R >= G)
                    %vertex = 'red';
                    vertex = [255,0,0];
                end
            end
        end
    end


% No.5 for RGBM
    if (mbvq == 'RGBM')
        %vertex = 'green';
        vertex = [0,255,0];
        if (R > G)
            if (R >= B)
                if (B < 0.5)
                    %vertex = 'red';
                    vertex = [255,0,0];
                else
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                end
            end
        end
        if (B > G)
            if (B >= R)
                if (R < 0.5)
                    %vertex = 'blue';
                    vertex = [0,0,255];
                else
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                end
            end
        end
    end


% No.6 for CMGB
    if (mbvq == 'CMGB')
        if (B > 0.5)
            if ( R > 0.5)
                if (G >= R)
                    %vertex = 'cyan';
                    vertex = [0,255,255];
                else
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                end
            else
                if (G > 0.5)
                    %vertex = 'cyan';
                    vertex = [0,255,255];
                else
                    %vertex = 'blue';
                    vertex = [0,0,255];
                end
            end
        else
            if ( R > 0.5)
                if (R - G + B >= 0.5)
                    %vertex = 'magenta';
                    vertex = [255,0,255];
                else
                    %vertex = 'green';
                    vertex = [0,255,0];
                end
            else
                if (G >= B)
                    %vertex = 'green';
                    vertex = [0,255,0];
                else
                    %vertex = 'blue';
                    vertex = [0,0,0];
                end
            end
        end
    end

end %function