
function sampling(rbm::RBM, vis; Patt=[], t_max=100)
  s_v = vis
  m_v = 0
  s_h = 0
  m_h = 0
  for t=1:t_max
    s_h,m_h = sample_hiddens01(rbm,s_v)
    s_v,m_v = sample_visibles01(rbm,s_h)
  end
  return s_v, m_v, s_h, m_h
end


function samplingRELU(rbm::RBM, vis::Mat; Patt=[], t_max=100)
  s_v = vis
  m_v = 0
  s_h = 0
  m_h = 0
  for t=1:t_max
    s_h,m_h = sample_hiddensRELU(rbm,s_v)
    s_v,m_v = sample_visibles01(rbm,s_h)
  end
  return s_v, m_v, s_h, m_h
end

function sample_visibles01(rbm::RBM, hid::Mat)
  m_v = sigm.(rbm.W'*hid .+ rbm.vbias)
  prob_1 = m_v
  s_v = float.((rand(Float32,size(prob_1)) .< prob_1))
  return s_v, m_v
end

function sample_hiddens01(rbm::RBM, vis::Mat)
  m_h = sigm.(rbm.W*vis .+ rbm.hbias) # = 1*p(1) -1*p(-1) = p(1) - (1-p(1)) = 2p(1)-1
  prob_1 = m_h
  s_h = float((rand(Float32,size(prob_1)) .< prob_1))
  return s_h,m_h
end

function sample_hiddensRELU(rbm::RBM, vis::Mat)
  act_h = rbm.W*vis .+ rbm.hbias
  idx_neg = findall(x->x .< -2, act_h)
  all_h = TruncatedNormal.(act_h,1,0,1000)
  m_h = mean.(all_h)
  s_h = rand.(all_h)
  m_h[idx_neg] .= 0
  s_h[idx_neg] .= 0
  return s_h, m_h
end

function meanFieldIte(rbm,vis; t_max=100)
  m_v = vis
  m_h = []
  for t=1:t_max
    _,m_h = sample_hiddens01(rbm,m_v)
    _,m_v = sample_visibles01(rbm,m_h)
  end
  return m_v, m_h
end

function meanFieldIteRELU(rbm,vis; t_max=100)
  m_v = vis
  m_h = []
  for t=1:t_max
    _,m_h = sample_hiddensRELU(rbm,m_v)
    _,m_v = sample_visibles01(rbm,m_h)
  end
  return m_v, m_h
end


function recst_err(rbm::RBM, vis::Matrix{Float64}; r=0.6, t_max=100)
  v_pos = vis
  mask = float(rand(Float32,size(v_pos)) .< (1-r))
  m_v = v_pos.*mask 
  id_MISS = findall(x->x==0,mask)
  m_v[id_MISS] .= 0.5


  # MF iterations
  for t=1:t_max
    if(rbm.ctx["mode_h"] == "Bernoulli")
        m_v[id_MISS] = sigm.(rbm.W'*sigm.(rbm.W*m_v .+ rbm.hbias) .+ rbm.vbias)[id_MISS]
    elseif(rbm.ctx["mode_h"] == "RELU")
        m_h = max.(0,rbm.W*m_v .+ rbm.hbias)
        m_v[id_MISS] = sigm.(rbm.W'*m_h)[id_MISS]
    end
  end

  return mean(sqrt.(sum(abs2,m_v.-v_pos,dims=1)))/sqrt(rbm.Nv*r)
end

function recst_smpl(rbm::RBM, vis::Matrix{Float64}, MISS::Matrix{Float64}; t_max=1000)
  m_v = vis.*(1 .- MISS)
  id_MISS = findall(x->x==1,MISS)
  m_v[id_MISS] = 0.5

  # MF iterations
  for t=1:t_max
    if(rbm.ctx["mode_h"] == "Bernoulli")
        m_v[id_MISS] = sigm.(rbm.W'*sigm.(rbm.W*m_v .+ rbm.hbias) .+ rbm.vbias)[id_MISS]
    elseif(rbm.ctx["mode_h"] == "RELU")
        m_h .= max(0,rbm.W*m_v .+ rbm.hbias)
        m_v[id_MISS] = sigm.(rbm.W'*m_h)[id_MISS]
    end            
  end

  return m_v
end

function recst_smplMCMC(rbm::RBM, vis::Matrix{Float64}, MISS::Matrix{Float64}; t_max=1000)
  m_v = vis.*(1 .-MISS)
  id_MISS = findall(x->x==1,MISS)
  id_FIX = findall(x->x==0,MISS) 
  m_v[id_MISS] = 0.5

  # MF iterations
  s_v = m_v
  m_v = 0
  for t=1:t_max
    if(rbm.ctx["mode_h"] == "Bernoulli")
        s_h,m_h = sample_hiddens01(rbm,s_v)
        s_v,m_v = sample_visibles01(rbm,s_h)
    elseif(rbm.ctx["mode_h"] == "RELU")
        s_h,m_h = sample_hiddensRELU(rbm,s_v)
        s_v,m_v = sample_visibles01(rbm,s_h)
    end
    s_v[id_FIX] = mv[id_FIX]
  end

  return s_v, m_v
end