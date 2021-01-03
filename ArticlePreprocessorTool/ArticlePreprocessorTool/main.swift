//
//  main.swift
//  ArticlePreprocessorTool
//
//  Created by Jonathan Duss on 02.05.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import Foundation
import ZaJoLibrary
import ArticleClassifierCore

var arguments = CommandLine.arguments

guard arguments.count == 3 else {
    fputs("usage: ArticlePreprocessorTool", stderr)
    exit(1)
}

let parameterInputFile = arguments[1]
let parameterOutputFile = arguments[2]

let processor = try ArticlePreprocessorTool(inputFilePath: parameterInputFile, outputFilePath: parameterOutputFile)
try processor.process()
