// Possible values for dtype
export const float32: number;
export const float64: number;
export const int32: number;

export interface ObjectTensor {
  data: Float32Array|Float64Array|Int32Array;
  shape: number[];
}

export class Tensor {
  static fromObject({data, shape}: ObjectTensor): Tensor;
  toObject(): ObjectTensor;
  toString(): string;
  cpu(): Tensor;
  cuda(): Tensor;
}

export class ScriptModule {
  constructor(path: string);
  forward(inputs: Tensor): any;
  forward(...inputs: Tensor[]): any;
  toString(): string;
  cpu(): ScriptModule;
  cuda(): ScriptModule;
  isCudaAvailable(): boolean;
}

export function rand(shape: number[], options?: {dtype?: number}): Tensor;
// The function actually support options at the end but TS can't express that
export function rand(...shape: number[] /**, options?: {dtype?: number} */):
    Tensor;
