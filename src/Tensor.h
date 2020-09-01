#pragma once

#include <memory>

#include <napi.h>
#include <torch/torch.h>

namespace torchjs
{

  class Tensor : public Napi::ObjectWrap<Tensor>
  {
  public:
    static Napi::Object Init(Napi::Env, Napi::Object exports);
    static Napi::Object FromTensor(Napi::Env, const torch::Tensor &);
    static bool IsInstance(Napi::Object &);
    static Napi::FunctionReference constructor;
    Tensor(const Napi::CallbackInfo &);

    torch::Tensor tensor();

  private:
    torch::Tensor tensor_;

    Napi::Value toString(const Napi::CallbackInfo &);
    Napi::Value toObject(const Napi::CallbackInfo &);
    static Napi::Value fromObject(const Napi::CallbackInfo &);
    Napi::Value cpu(const Napi::CallbackInfo &info);
    Napi::Value cuda(const Napi::CallbackInfo &info);
  };

} // namespace torchjs