function entropy_bin(x)
  return -x.*log.(x)
end


function sigm(x)
  return 1/(1 .+ exp.(-x))
end



function plotPCA(rbm, X; fname="test.png")
  println("plotting...")
  vis = rand(Nv,250)

  idp = randperm(size(X,2))

  sv,_,_,_ = sampling(rbm,vis;t_max=200)
  sv_X,_,_,_ = sampling(rbm,X[:,idp[1:250]];t_max=200)
  m_v,_ = meanFieldIte(rbm,sv;t_max=100)
  m_v_X,_ = meanFieldIte(rbm,sv_X;t_max=100)

  u,s,v = svd(rbm.W)

  f,ax = plt[:subplots](1,3)
  ax[1][:scatter](X'*v[:,1],X'*v[:,2], s=5)
  ax[1][:scatter](sv'*v[:,1],sv'*v[:,2], s=5)
  ax[1][:scatter](sv_X'*v[:,1],sv_X'*v[:,2], s=5)
  ax[1][:scatter](m_v'*v[:,1],m_v'*v[:,2], s=5)
  ax[1][:scatter](m_v_X'*v[:,1],m_v_X'*v[:,2], s=5)
  # ax[1,1][:scatter](mv'*v[:,1],mv'*v[:,2])

  ax[2][:scatter](X'*v[:,3],X'*v[:,4], s=5)
  ax[2][:scatter](sv'*v[:,3],sv'*v[:,4], s=5)
  ax[2][:scatter](sv_X'*v[:,3],sv_X'*v[:,4], s=5)
  ax[2][:scatter](m_v'*v[:,3],m_v'*v[:,4], s=5)
  ax[2][:scatter](m_v_X'*v[:,3],m_v_X'*v[:,4], s=5)
  # ax[1,2][:scatter](mv'*v[:,3],mv'*v[:,4])

  ax[3][:scatter](X'*v[:,5],X'*v[:,6], s=5)
  ax[3][:scatter](sv'*v[:,5],sv'*v[:,6], s=5)
  ax[3][:scatter](sv_X'*v[:,5],sv_X'*v[:,6], s=5)
  ax[3][:scatter](m_v'*v[:,5],m_v'*v[:,6], s=5)
  ax[3][:scatter](m_v_X'*v[:,5],m_v_X'*v[:,6], s=5)
  # ax[1,3][:scatter](mv'*v[:,5],mv'*v[:,6])
  f[:savefig](fname)

end

function patch10(im)
  si = Int(sqrt(size(im,1)))
  im_p = zeros(10*si,10*si)
  idx = 1
  for i=1:10 
    for j=1:10
      im_p[(i-1)*si+1:i*si,(j-1)*si+1:j*si] = reshape(im[:,idx],28,28)
      idx += 1
    end
  end
  return im_p
end






function uniquetol_vec(rbm::RBM, itr::Array{T,2}; tol=1e-6) where {T<:Real}
  _itr = copy(itr)
  out = [ ];
  change = true
  while(length(_itr)>0)
    # push!(out, Array{T,2}(_itr[:,1]))
    idxs = findall(dropdims(sum(abs2,(itr .- _itr[:,1]),dims=1),dims=1)./size(itr,1) .< tol)
    ## idx = find( free_energy_tap(rbm,))
    ##println((sum(abs2,(itr .- _itr[:,1]),1))[idxs])
    ##println("idsx ",idxs)
    keep_idxs = findall(dropdims(sum(abs2,(_itr .- _itr[:,1]),dims=1),dims=1)./size(itr,1) .> tol)
    ##println(keep_idxs)
    _itr = _itr[:,keep_idxs]
    #deleteat!(_itr,remove_idxs)
    #if(length(remove_idxs)==0)
    #  change=false
    #end
    if(length(idxs)>1)
      push!(out,idxs[1])
    end
  end
  if(isempty(out))
    out = collect(1:1:size(itr,2))
  end

  return out
end

function saveWeights(rbm,fname)
  h5write(fname,"W",rbm.W)
  h5write(fname,"hbias",rbm.hbias)
  h5write(fname,"vbias",rbm.vbias)
end

