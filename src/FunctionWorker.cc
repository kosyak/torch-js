#include "FunctionWorker.h"

namespace torchjs
{
  FunctionWorker::FunctionWorker(Napi::Env env, std::function<c10::IValue()> _workFunction, std::function<Napi::Value(Napi::Env, c10::IValue)> _postWorkFunction)
      : promise(Napi::Promise::Deferred::New(env)), workFunction(_workFunction), postWorkFunction(_postWorkFunction), AsyncWorker(env) {}

  void FunctionWorker::Execute()
  {
    value = workFunction();
  }

  void FunctionWorker::OnOK()
  {
    auto result = postWorkFunction(Env(), value);
    promise.Resolve(result);
  }

  void FunctionWorker::OnError(const Napi::Error &e)
  {
    promise.Reject(e.Value());
  }

  Napi::Promise FunctionWorker::GetPromise()
  {
    return promise.Promise();
  }

} // namespace torchjs