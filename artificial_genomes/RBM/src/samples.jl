
function sampling(rbm::RBM, vis::Matrix{Float64}; Patt=[], t_max=100)
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


function samplingRELU(rbm::RBM, vis::Matrix{Float64}; Patt=[], t_max=100)
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

function sample_visibles01(rbm::RBM, hid::Matrix{Float64})
  m_v = sigm.(rbm.W'*hid .+ rbm.vbias)
  prob_1 = m_v
  s_v = float.((rand(size(prob_1)) .< prob_1))
  return s_v, m_v
end

function sample_hiddens01(rbm::RBM, vis::Matrix{Float64})
  m_h = sigm.(rbm.W*vis .+ rbm.hbias) # = 1*p(1) -1*p(-1) = p(1) - (1-p(1)) = 2p(1)-1
  prob_1 = m_h
  s_h = float((rand(size(prob_1)) .< prob_1))
  return s_h,m_h
end

function sample_hiddensRELU(rbm::RBM, vis::Matrix{Float64})
  act_h = rbm.W*vis .+ rbm.hbias
  idx_neg = find(x->x .< -2, act_h)
  all_h = TruncatedNormal.(act_h,1,0,1000)
  m_h = mean.(all_h)
  s_h = rand.(all_h)
  m_h[idx_neg] = 0
  s_h[idx_neg] = 0
  return s_h, m_h
end