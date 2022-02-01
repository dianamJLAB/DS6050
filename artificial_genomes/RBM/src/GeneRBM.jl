
using Distributions
# using Base.LinAlg.BLAS
using HDF5
using LinearAlgebra
using Random

Mat{T} = AbstractArray{T,2}
Vec{T} = AbstractArray{T, 1}


mutable struct RBM
  W::Matrix{Float64}
  Ws::Matrix{Int}
  W2::Matrix{Float64}
  hbias::Vector{Float64}
  vbias::Vector{Float64}
  dW::Matrix{Float64}
  dW_prev::Matrix{Float64}
  momentum::Float64
  Nv::Int
  Nh::Int
  p_contdiv::Matrix{Float64}
  fe_tr
  pl_tr
  s_tr
  ctx::Dict{Any,Any}
  err_recst
  err_recst_t  
end

include("utils.jl")
include("scores.jl")
include("sampling.jl")
include("update.jl")

function RBM(n_vis::Int, n_hid::Int; sigma=0.01, momentum=0.0, seed=10)
  # srand(seed)
  Random.seed!(seed)
  println("=====================================")
  println("RBM parameters")
  println("=====================================")
  println("  + Sigma vis-weights:              $sigma")
  RBM(rand(Normal(0, sigma), n_hid, n_vis),
    rand(Bernoulli(0.1),n_hid,n_vis),
    zeros(n_hid,n_vis),
    zeros(n_hid),
    zeros(n_vis),
    #rand(Normal(0,0.1),n_hid),
    zeros(n_hid, n_vis),
    zeros(n_hid, n_vis),
    momentum,
    n_vis,
    n_hid,
    zeros(0,0),
    [],
    [],
    [],  
    Dict{Any,Any}("pcd"=>false, "WriteSmpl"=>1e10, "fnameParam"=>"tmp", "WriteWeights"=>1e10,
                  "fnameSmpl"=>"tmp","upd_hbias"=>true, "mode_h"=>"Bernoulli",
                  "upd_vbias"=>true, "upd_W"=>true, "dp_init"=>false, "SaveSC"=>1e10),
    [],
    [])
end

function setPCD!(rbm::RBM)
  rbm.ctx["pcd"] = true
end

function setWriteSmpl!(rbm::RBM, dt, fname)
  rbm.ctx["WriteSmpl"] = dt
  rbm.ctx["fnameSmpl"] = fname
end

function setWriteWeights!(rbm::RBM, dt, fname)
  rbm.ctx["WriteWeights"] = dt
  rbm.ctx["fnameParam"] = fname
end

function setHBIAS!(rbm::RBM, act)
  rbm.ctx["upd_hbias"] = act
end

function setVBIAS!(rbm::RBM, act)
  rbm.ctx["upd_vbias"] = act
end

function setModeH!(rbm::RBM, mode)
  rbm.ctx["mode_h"] = mode
end

function setW!(rbm::RBM, mode)
  rbm.ctx["upd_W"] = mode
end

function setSaveSc!(rbm::RBM, dt)
  rbm.ctx["SaveSc"] = dt
end

function initBiasFromSamples(rbm,Data; eps=1e-8)
  InitialVisBias = zeros(size(Data,1))
  if !isempty(Data)
    ProbVis = mean(Data,dims=2)             # Mean across  samples

    ProbVis = max.(ProbVis,eps)              # Some regularization (avoid Inf/NaN)
    ProbVis = min.(ProbVis,1 .- eps)          # ''

    InitialVisBias = log.(ProbVis ./ (1 .- ProbVis)) # Biasing as the log-proportion
  end
  rbm.vbias = InitialVisBias[:,1]
end

function fit(rbm::RBM, X::Mat{Float64}; lr=0.01, n_iter=10, batch_size=50, n_gibbs=10, damp=0.5, t_max=100, fname="", vscatter=false, Patt=[], n_pcd=100, δt=100000, Xtest = [], y=[])
  nv = size(rbm.W,2)
  nh = size(rbm.W,1)
  Ntot = nv+nh

  n_samples = size(X,2)
  n_batches = round(Int,ceil(n_samples / batch_size))

  rbm.W2 = abs2.(rbm.W)

  println("=====================================")
  println("RBM Training")
  println("=====================================")
  println("  + Training Samples:   $n_samples")
  println("  + Features:           $nv")
  println("  + Hidden Units:       $nh")
  println("  + Epochs to run:      $n_iter")
  println("  + Learning rate:      $lr")
  println("  + Gibbs Steps:        $n_gibbs")
  println("=====================================")

  # put persistent cd if present  
  if(rbm.ctx["pcd"])
    rbm.p_contdiv = float(rand(size(X,1),n_pcd) .< 0.5)
    rbm.p_contdiv,_,_,_ = sampling(rbm,rbm.p_contdiv;t_max=200)
  end

  for itr=1:n_iter

    if(rbm.ctx["dp_init"])
      rbm.p_contdiv[:,Int(n_pcd/2+1):n_pcd] = X[:,randperm(size(X,2))[1:Int(n_pcd/2)]]
      rbm.p_contdiv,_,_,_ = sampling(rbm,rbm.p_contdiv; t_max=200)
    end

    ind_p = randperm(size(X,2)) # permutations of the dataset
    ## ind_invp = invperm(ind_p) # to recover the correct order if
    Xp = X[:,ind_p]

    println("itr $itr")
    for i=1:n_batches
      batch = Xp[:,((i-1)*batch_size + 1):min(i*batch_size, size(Xp,2))]
      # batch = full(batch)

      fit_batch!(rbm,batch,lr; t_max=t_max,it=itr,itmb=i)        
      
      if((i-1)%δt==0)
        err02 = recst_err(rbm,X[:,1:100]; r=0.2)        
        err06 = recst_err(rbm,X[:,1:100]; r=0.6)
        err10 = recst_err(rbm,X[:,1:100]; r=1)

        if(!isempty(Xtest))
          err02_t = recst_err(rbm,Xtest[:,1:100]; r=0.2)
          err04_t = recst_err(rbm,Xtest[:,1:100]; r=0.4)
          err06_t = recst_err(rbm,Xtest[:,1:100]; r=0.6)
          err08_t = recst_err(rbm,Xtest[:,1:100]; r=0.8)
          err10_t = recst_err(rbm,Xtest[:,1:100]; r=1.0)
        end

        push!(rbm.err_recst,[err02,err06,err10])
        u,s,v = svd(rbm.W)
        push!(rbm.s_tr,s)

        print(size(findall(x->x>1,s))," ")
        println(err02," ", err06," ", err10," ",s[1:5])
        if(!isempty(Xtest))
          push!(rbm.err_recst_t,[err02_t,err04_t,err06_t,err08_t,err10_t])
          println(err02_t," ", err06_t," ", err10_t)
        end

        fe_tap_rdm = likelihood_tap(rbm,X)
        fe_tap_data = likelihood_tap(rbm,X;data_start=true)
        push!(rbm.fe_tr,fe_tap_rdm)
        println("# Likelihood (rdm start)= ",fe_tap_rdm)
        println("# Likelihood (data start)= ",fe_tap_data)
        
        s_v,_,_,_ = sampling(rbm,rand(rbm.Nv,1000);t_max=500)        
        PL_rdm = pseudo_likelihood(rbm,s_v)
        PL_data = pseudo_likelihood(rbm,X)
        push!(rbm.pl_tr,PL_data)
        println("# PLikelihood (rdm start)=",PL_rdm)
        println("# PLikelihood (data start)=",PL_data)
      end      
    end

    if(!isempty(y))
      err = compute_classif_error_r(rbm::RBM, X, y;r=10)
      println("Itr=",itr," ClassifErr=",err)
    end

    
    # sample generated samples 
    if(itr%rbm.ctx["WriteSmpl"] == 0)
      m_v = []
      if(rbm.ctx["mode_h"]=="Bernoulli")
        s_v,m_v,_,_ =sampling(rbm,rand(784,100);t_max=1000) 
      elseif (rbm.ctx["mode_h"]=="RELU")
        s_v,m_v,_,_ =samplingRELU(rbm,rand(784,100);t_max=1000) 
      end
      p10=patch10(m_v)
      plt[:imshow](p10)
      plt[:savefig](string("smpl_",rbm.ctx["fnameSmpl"],"_itr",itr,".png"))
      p10=patch10(s_v)
      plt[:imshow](p10)
      plt[:savefig](string("smplS_",rbm.ctx["fnameSmpl"],"_itr",itr,".png"))
    end

    # save weights
    if(itr%rbm.ctx["WriteWeights"] == 0)
      saveWeights(rbm,string("model_",rbm.ctx["fnameParam"],"_itr",itr,".d"))
    end

    # save scatter
    if(itr%rbm.ctx["SaveSc"] == 0)      
      u,s,v = svd(rbm.W)
      if(rbm.ctx["mode_h"] == "Bernoulli")                    
          s_v,_,_,_ = sampling(rbm,rand(Nv,1000);t_max=500)
          m_v,_ = meanFieldIte(rbm,rbm.p_contdiv;t_max=200)                    
      elseif(rbm.ctx["mode_h"] == "RELU")
          s_v,_,_,_ = samplingRELU(rbm,rand(Nv,1000);t_max=500)                        
          m_v,_ = meanFieldIteRELU(rbm,rbm.p_contdiv;t_max=200)
      end
      f,ax = plt[:subplots](2,2)
      ax[1,1][:hist2d](X'*v[:,1],X'*v[:,2],bins=50)
      ax[1,1][:scatter](rbm.p_contdiv'*v[:,1],rbm.p_contdiv'*v[:,2],color="red",s=2)
      ax[1,1][:scatter](m_v'*v[:,1],m_v'*v[:,2],color="green",s=2)
      ax[1,2][:hist2d](s_v'*v[:,1],s_v'*v[:,2],bins=50)
      ax[1,2][:scatter](rbm.p_contdiv'*v[:,1],rbm.p_contdiv'*v[:,2],color="red",s=2)
      ax[1,2][:scatter](m_v'*v[:,1],m_v'*v[:,2],color="green",s=2)
      # ax[1][:scatter](m_v'*v[:,1],m_v'*v[:,2])

      ax[2,1][:hist2d](X'*v[:,3],X'*v[:,4],bins=50)
      ax[2,1][:scatter](rbm.p_contdiv'*v[:,3],rbm.p_contdiv'*v[:,4],color="red",s=2)
      ax[2,1][:scatter](m_v'*v[:,3],m_v'*v[:,4],color="green",s=2)
      ax[2,2][:hist2d](s_v'*v[:,3],s_v'*v[:,4],bins=50)
      ax[2,2][:scatter](rbm.p_contdiv'*v[:,3],rbm.p_contdiv'*v[:,4],color="red",s=2)  
      ax[2,2][:scatter](m_v'*v[:,3],m_v'*v[:,4],color="green",s=2)
      # ax[2][:scatter](m_v'*v[:,3],m_v'*v[:,4])
                
      f[:savefig](string("sc_",rbm.ctx["fnameParam"],"_itr",itr,".png"))
    end

  end
end





function fit_batch!(rbm::RBM, vis::Mat{Float64}, lr;
                    n_gibbs=10, damp=0.0, t_max=100, it=0, itmb=0)

    v_pos = []; h_pos = []
    v_neg = []; h_neg = []

    if(rbm.ctx["mode_h"] == "Bernoulli")
      v_pos, h_pos = getPosTerm01(rbm,vis)
      v_neg, h_neg = getNegTermMCMC01(rbm,vis;t_max=t_max)
    elseif(rbm.ctx["mode_h"] == "RELU")
      v_pos, h_pos = getPosTermRELU(rbm,vis)
      v_neg, h_neg = getNegTermMCMCRELU(rbm,vis;t_max=t_max)
    end

    learn_rate = lr
    update_weights(rbm,v_pos,h_pos,v_neg,h_neg,learn_rate)    
end



