'use strict';

var path = require('path');
var bindings = require('bindings');
var moduleRoot = bindings.getRoot(bindings.getFileName());
var buildFolder = path.join(moduleRoot, 'build', 'Release');
process.env['path'] = buildFolder + ';' + process.env['path'];

var torch = require('bindings')('torch-js');
module.exports = torch;