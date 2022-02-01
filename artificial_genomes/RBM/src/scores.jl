
using SpecialFunctions

function compute_classif_error_r(rbm::RBM, X, y; Nx=1000, r=10)  
  NL = size(unique(y),1)
  
  indLow = rbm.Nv+1-NL*r
  Nd = indLow-1
  ind_p = randperm(size(X,2))
  Xt = X[:,ind_p[1:Nx]]

  MISS = zeros(size(Xt))
  MISS[indLow:end,:] = 1

  m_v = recst_smpl(rbm,Xt,MISS; t_max=100)

  num_cat=10
  succ = 0
  for i=1:size(m_v,2)
    idx_max = 0
    vmax=0  
    for c=0:(num_cat-1)
      tmax = 0
      for ir=1:r
        tmax += m_v[Nd+c*num_cat+ir,i]
      end
      if(tmax>vmax)
        vmax=tmax
        idx_max=c
      end
    end
    if(y[ind_p[i]]==idx_max)
      succ += 1
    end
  end

  return succ/size(m_v,2)
end

function compute_classif_error(rbm::RBM, X, y; Nx=1000)
  NL = size(unique(y),1)
  indLow = rbm.Nv+1-NL
  ind_p = randperm(size(X,2))
  Xt = X[:,ind_p[1:Nx]]
  MISS = zeros(size(Xt))
  MISS[indLow:end,:] = 1
  mv = recst_smpl(rbm,Xt,MISS; t_max=100)

  succ = 0
  id_missed = []
  for i=1:size(mv,2)
    if(y[ind_p[i]] == (indmax(mv[indLow:end,i])-1))
      succ += 1
    else 
      push!(id_missed,i)
    end
  end

  return succ/size(mv,2)
end

function compute_confusion_mat(rbm::RBM, X, y; Nx=1000)
  NL = size(unique(y),1)
  indLow = rbm.Nv+1-NL
  ind_p = randperm(size(X,2))
  Xt = X[:,ind_p[1:Nx]]
  MISS = zeros(size(Xt))
  MISS[indLow:end,:] = 1
  mv = recst_smpl(rbm,Xt,MISS; t_max=100)

  conf_mat = zeros(NL,NL)
  succ = 0
  id_missed = []
  for i=1:size(mv,2)
    conf_mat[Int(y[ind_p[i]]+1),Int(indmax(mv[indLow:end,i]))] += 1
    if(y[ind_p[i]] == (indmax(mv[indLow:end,i])-1))
      succ += 1
    else 
      push!(id_missed,i)
    end
  end

  return conf_mat
end

function free_energy(rbm::RBM, vis::Mat)
    vb = sum(vis .* rbm.vbias, dims=1)
    fe_exp = 1 .+ exp.(rbm.W * vis .+ rbm.hbias)
    # tofinite!(fe_exp; nozeros=true)

    Wx_b_log = sum(log.(fe_exp), dims=1)
    result = - vb - Wx_b_log

    return result
end

function free_energyRELU(rbm::RBM, vis::Mat)
    vb = sum(vis .* rbm.vbias, dims=1)
    act_h = rbm.W * vis .+ rbm.hbias
    # indNeg = findall(x->x.<-2,act_h)
    fe_exp = 0.5 .+ 0.5*erf.(act_h.*sqrt(0.5))
    fe_exp .= max.(1e-5,fe_exp)
            
    Wx_b_log = sum(log.(fe_exp) .+ 0.5*act_h.^2, dims=1)
    result = - vb - Wx_b_log

    return result
end

function likelihood_tap(rbm::RBM, vis; data_start=false)    
    s_v = rand(rbm.Nv,100)
    if(data_start)
      idp = randperm(100)
      s_v = X[:,idp]
    end
    m_v = 0
    m_h = 0
    if(rbm.ctx["mode_h"]=="Bernoulli")
        _,m_v,_,_ = sampling(rbm,s_v;t_max=200)
        m_v,m_h = meanFieldIte(rbm,m_v;t_max=100)
    elseif(rbm.ctx["mode_h"]=="RELU")
        _,m_v,_,_ = samplingRELU(rbm,s_v;t_max=200)
        m_v,m_h = meanFieldIteRELU(rbm,m_v;t_max=100)
    end
    all_pf = uniquetol_vec(rbm,free_energy_tap(rbm,m_v,m_h);tol=0.0000001)  # find duplicates

    Ftap = -free_energy_tap(rbm,m_v[:,all_pf],m_h[:,all_pf])*rbm.Nv
    max_tap = maximum(Ftap)
    Ftap = Ftap .- max_tap
    Z = sum(exp.(Ftap))
    Ftap = -(log(Z) + max_tap)/rbm.Nv
    # Ftap = -log(sum(exp.(-rbm.Nv*free_energy_tap(rbm,m_v[:,all_pf],m_h[:,all_pf]))))/rbm.Nv

    LLData = 0
    if(rbm.ctx["mode_h"]=="Bernoulli")
        LLData = - mean(free_energy(rbm,vis),dims=2)/rbm.Nv
    elseif(rbm.ctx["mode_h"]=="RELU")
        LLData = - mean(free_energyRELU(rbm,vis),dims=2)/rbm.Nv
    end
    #vb = sum(vis .* rbm.vbias, dims=1)
    #fe_exp = 1 .+ exp.(rbm.W * vis .+ rbm.hbias)
    #LLData = mean(vb .+ sum(log.(fe_exp),dims=1))/rbm.Nv

    return (LLData .+ Ftap)[1]
end



function free_energy_tap(rbm::RBM, mag_vis::Mat, mag_hid::Mat) 
    mag_vis2 = abs2.(mag_vis)
    mag_hid2 = abs2.(mag_hid)

    Entropy = 0
    if(rbm.ctx["mode_h"]=="Bernoulli")
        Entropy = entropyMF(rbm, mag_hid) + entropyMF(rbm, mag_vis) # S
    elseif(rbm.ctx["mode_h"]=="RELU")
        Entropy = entropyTGauss(rbm, mag_hid) + entropyMF(rbm, mag_vis) # S
    end
    U_naive = -( sum(mag_vis.*rbm.vbias,dims=1) + sum(mag_hid.*rbm.hbias,dims=1) + sum(mag_hid.*(rbm.W*mag_vis),dims=1) ) # E
    Onsager = 0 # ( 0.5*sum( (mag_hid - mag_hid2) .* (rbm.W2 * (mag_vis - mag_vis2)),1) )
    # Onsager = -( 0.5* sum( (rbm.W2*σi).*(1-mag_hid2),1 ) )
    ## Onsager = -( 0.5* sum( (W2*(1-mag_vis2)).*(1-mag_hid2),1 ) )/rbm.Nv

    fe_tap = U_naive .+ Onsager .- Entropy
    return fe_tap./rbm.Nv
end


function entropyMF(rbm::RBM, mag::Mat)
  S = sum(entropy_bin(mag) + entropy_bin(1 .- mag),dims=1)
  return S
end
        
function entropyTGauss(rbm::RBM, mag::Mat)
  S = sum(- mag.*mag*0.5, dims=1) # .+ 0.5*log.(2*π),dims=1)
  return S
end
        
function pseudo_likelihood(rbm::RBM, vis::Mat; sample_size=10000)
            
    n_feat, n_samples = size(vis)
    vis_corrupted = copy(vis)
    idxs = rand(1:n_feat, n_samples)
    for (i, j) in zip(idxs, 1:n_samples)
        vis_corrupted[i, j] = 1 .- vis_corrupted[i, j] 
    end

    if(rbm.ctx["mode_h"] == "Bernoulli")
        fe = free_energy(rbm, vis)
        fe_corrupted = free_energy(rbm, vis_corrupted)
    elseif(rbm.ctx["mode_h"] == "RELU")
        fe = free_energyRELU(rbm, vis)
        fe_corrupted = free_energyRELU(rbm, vis_corrupted)
    end
    fe_diff = fe_corrupted - fe
    # tofinite!(fe_diff; nozeros=true)
    score_row =  n_feat * log.(sigm.(fe_diff))

    result = map(Float64, dropdims(score_row', dims=2))

    return mean(result)/rbm.Nv
end