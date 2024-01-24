function [Output,r] = GetOutput_PBI(varargin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    selfadapt = true;
    if nargin == 3
        selfadapt = false;
        delt      = varargin{3};
    end
    
    Pop = varargin{1};
    Ref = varargin{2};
    
    if selfadapt
        delt_l = -20;
        delt_u = 20;
        r = 0;     
        while r>0.7 || r<0.3
            delt_c = (delt_l + delt_u)/2;
            if abs(delt_l-delt_u)<1e-1
                break;
            end
            [l,r] = split_data(Pop,Ref,delt_c);
            if r > 0.7
               delt_l = delt_c;
            elseif r < 0.3
               delt_u = delt_c;
            end
        end
    else
        [l,~] = split_data(Pop,Ref,delt);
    end
    Output = l;
end

function [Output,rate] = split_data(Pop,Ref,delt)
    N      = size(Pop,1);
    popind = 1 : N;
    Output = true(N,1);
    [~,ref_index] = max(1-pdist2(Pop,Ref,'cosine'),[],2);%求与每个解角度差最小的参考解的编号
    Z = min(Pop,[],1);
    for i = 1 : size(Ref,1)
        sub_pop    = Pop(ref_index==i,:);  %选择i号参考解的所有解
        sub_popind = popind(ref_index==i);  %选择i号参考解的所有解编号
        BOUND      = Ref(i,:);
        w = BOUND-Z;
        W = w./sqrt(sum((w).^2,2)); %参考向量-Z的cos值
        normW   = sqrt(sum((W).^2,2)); %W的模(这不是必定为1？
        normP   = sqrt(sum((sub_pop-repmat(Z,size(sub_pop,1),1)).^2,2)); %每个点离Z的距离
        normR   = sqrt(sum((BOUND-Z).^2,2));  %参考点离Z的距离
        CosineP = (sum((sub_pop-repmat(Z,size(sub_pop,1),1)).*repmat(W,size(sub_pop,1),1),2)./normW./normP)-1e-6; 
        %以Z为角点每个点和参考点夹角的cos值
        g = normP.*CosineP + delt*normP.*sqrt(1-CosineP.^2);
        k = normR;
        g = g./k; %g=cos∠PZR+delt*sin∠PZR
        Output(sub_popind(g>1)) = false;
    end
	rate = sum(Output == 1)/length(Output);
end