#include "ScriptModule.h"

#include "Tensor.h"
#include "utils.h"

namespace torchjs
{
  Napi::Object ScriptModule::Init(Napi::Env env, Napi::Object exports)
  {
    Napi::Function func = DefineClass(env, "ScriptModule",
                                      {InstanceMethod("forward", &ScriptModule::forward),
                                       InstanceMethod("toString", &ScriptModule::toString),
                                       InstanceMethod("cpu", &ScriptModule::cpu),
                                       InstanceMethod("cuda", &ScriptModule::cuda),
                                       StaticMethod("isCudaAvailable", &ScriptModule::isCudaAvailable)});

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("ScriptModule", func);
    return exports;
  }

  ScriptModule::ScriptModule(const Napi::CallbackInfo &info) : Napi::ObjectWrap<ScriptModule>(info)
  {
    Napi::HandleScope scope(info.Env());
    Napi::String value = info[0].As<Napi::String>();
    path_ = value;
    module_ = torch::jit::load(value);
  }

  Napi::FunctionReference ScriptModule::constructor;

  Napi::Value ScriptModule::forward(const Napi::CallbackInfo &info)
  {
    Napi::EscapableHandleScope scope(info.Env());
    c10::IValue *outputs;
    module_.eval();

    // input: Tensor[]
    // TODO: Support other type of IValue, e.g., list
    auto len = info.Length();
    std::vector<torch::jit::IValue> inputs;
    for (size_t i = 0; i < len; ++i)
    {
      Tensor *tensor = Napi::ObjectWrap<Tensor>::Unwrap(info[i].As<Napi::Object>());
      inputs.push_back(tensor->tensor());
    }
    outputs = &module_.forward(inputs);
    return scope.Escape(deRefIValue(info.Env(), *outputs));
  }

  Napi::Value ScriptModule::deRefIValue(Napi::Env env, const c10::IValue &iValue)
  {
    if (iValue.isTensor())
    {
      return Tensor::FromTensor(env, iValue.toTensor());
    }
    else if (iValue.isList())
    {
      auto list = iValue.toList();
      auto jsList = Napi::Array::New(env);
      for (auto i = 0; i < list.size(); i++)
      {
        jsList[i] = deRefIValue(env, list[i]);
      }
      return jsList;
    }
    else if (iValue.isGenericDict())
    {
      auto dict = iValue.toGenericDict();
      auto jsDict = Napi::Object::New(env);
      for (auto iter = dict.begin(); iter != dict.end(); iter++)
      {
        auto key = deRefIValue(env, iter->key());
        auto value = deRefIValue(env, iter->value());
        jsDict.Set(key, value);
      }
      return jsDict;
    }
    else if (iValue.isInt())
    {
      return Napi::Number::New(env, iValue.toInt());
    }
    else if (iValue.isDouble())
    {
      return Napi::Number::New(env, iValue.toDouble());
    }
    else if (iValue.isBool())
    {
      return Napi::Boolean::New(env, iValue.toBool());
    }
    else if (iValue.isString())
    {
      return Napi::String::New(env, iValue.toString().get()->string());
    }
    else if (iValue.isTuple())
    {
      auto list = iValue.toTuple()->elements();
      auto jsList = Napi::Array::New(env);
      for (auto i = 0; i < list.size(); i++)
      {
        jsList[i] = deRefIValue(env, list[i]);
      }
      return jsList;
    }
    throw Napi::Error::New(env, "Unsupported output type from ScriptModule");
  }

  Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info)
  {
    return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
  }

  Napi::Value ScriptModule::cpu(const Napi::CallbackInfo &info)
  {
    module_.to(at::kCPU);
    return info.This();
  }

  Napi::Value ScriptModule::cuda(const Napi::CallbackInfo &info)
  {
    if (torch::cuda::is_available())
    {
      module_.to(at::kCUDA);
    }
    else
    {
      throw Napi::Error::New(info.Env(), "CUDA is not aviliable");
    }
    return info.This();
  }

  Napi::Value ScriptModule::isCudaAvailable(const Napi::CallbackInfo &info)
  {
    return Napi::Boolean::New(info.Env(), torch::cuda::is_available());
  }

} // namespace torchjs