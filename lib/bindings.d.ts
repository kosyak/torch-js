type bindingFunction = (mod: string) => any;
interface bindingInterface extends bindingFunction {
  getRoot: (file: string) => string;
  getFileName: (calling_file?: string) => string;
}
declare var bindings: bindingInterface;

declare module 'bindings' {
  export = bindings;
}
