

function update_weights(rbm,v_pos,h_pos,v_neg,h_neg,lr)
  rbm.dW .= lr.*( (h_pos*v_pos')./size(v_pos,2) .- (h_neg*v_neg')./size(v_neg,2) ) # /sqrt(rbm.Nv)
  if(rbm.ctx["upd_W"])
    rbm.W .+= rbm.dW
  end
  rbm.W2 = abs2.(rbm.W)
  # copy!(rbm.dW_prev,rbm.dW)

    
  if(rbm.ctx["upd_vbias"])
    rbm.vbias .+= lr.*dropdims(sum(v_pos,dims=2)./size(v_pos,2)-sum(v_neg,dims=2)./size(v_neg,2),dims=2)
  end
  if(rbm.ctx["upd_hbias"])
    rbm.hbias .+= lr.*dropdims(sum(h_pos,dims=2)./size(v_pos,2)-sum(h_neg,dims=2)./size(v_neg,2),dims=2)
  end
end


function getPosTerm01(rbm::RBM, vis::Matrix{Float64})
  return vis,sigm.(rbm.W*vis .+ rbm.hbias)
end

function getPosTermRELU(rbm::RBM, vis::Matrix{Float64})
  mμ = rbm.W*vis .+ rbm.hbias
  idx_neg = findall(x->x .< -2, mμ)
  m_h = mean.(TruncatedNormal.(mμ,1,0,1000))
  m_h[idx_neg] .= 0
  return vis,m_h
  # return vis,mean(TruncatedNormal(mμ,1,0,1000))
end

function getNegTermMCMC01(rbm::RBM, vis::Matrix{Float64}; t_max=100)
  n_vis = vis
  if(rbm.ctx["pcd"])
    n_vis = rbm.p_contdiv
  end

  s_h, m_h = sample_hiddens01(rbm,n_vis)

  s_v, m_v = sample_visibles01(rbm,s_h)
  s_h, m_h = sample_hiddens01(rbm,s_v)
  for t=1:(t_max-1)
    s_v, m_v = sample_visibles01(rbm,s_h)
    s_h, m_h = sample_hiddens01(rbm,s_v)
  end

  if(rbm.ctx["pcd"])
    rbm.p_contdiv = s_v
  end
  return s_v, s_h
end

function getNegTermMCMCRELU(rbm::RBM, vis::Matrix{Float64}; t_max=100)
  n_vis = vis
  if(rbm.ctx["pcd"])
    n_vis = rbm.p_contdiv
  end

  s_h, m_h = sample_hiddensRELU(rbm,n_vis)

  s_v, m_v = sample_visibles01(rbm,s_h)
  s_h, m_h = sample_hiddensRELU(rbm,s_v)
  for t=1:(t_max-1)
    s_v, m_v = sample_visibles01(rbm,s_h)
    s_h, m_h = sample_hiddensRELU(rbm,s_v)
  end

  if(rbm.ctx["pcd"])
    rbm.p_contdiv = copy(s_v)
  end
  return s_v, s_h
end