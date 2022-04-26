type BindingFunction = (mod: string) => any;
interface BindingInterface extends BindingFunction {
  getRoot: (file: string) => string;
  getFileName: (callingFile?: string) => string;
}
declare let bindings: BindingInterface;

declare module 'bindings' {
  export = bindings;
}
