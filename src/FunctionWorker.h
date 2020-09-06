#pragma once

#include <napi.h>
#include <functional>
#include "Tensor.h"

namespace torchjs
{
  class FunctionWorker : public Napi::AsyncWorker
  {
  public:
    FunctionWorker(Napi::Env env, std::function<c10::IValue()> _workFunction, std::function<Napi::Value(Napi::Env, c10::IValue)> _postWorkFunction);
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error &e) override;
    Napi::Promise GetPromise();

  private:
    Napi::Promise::Deferred promise;
    std::function<c10::IValue()> workFunction;
    std::function<Napi::Value(Napi::Env, c10::IValue)> postWorkFunction;
    c10::IValue value;
  };
} // namespace torchjs