//
//  LSA.swift
//  ArticleClassifierCore-macos
//
//  Created by Jonathan Duss on 05.04.19.
//

import Foundation

//class LSA {
//
//    private let documents: [String]
//    let tfIdf: TfIdf
//
//    init(document: [String]) {
//        self.documents = document.map({$0.lowercased()})
//        self.tfIdf = TfIdf(texts: documents)
//    }
//
//    private func generateTfIdfMatrix() -> Matrix {
//
//        let matrix = Matrix(tfIdf.allTerms.count, documents.count)
//
//        for col in 0..<documents.count {
//            var tfidfVector = tfIdf.tfIdfVector(text: documents[col])
//            for row in 0..<tfIdf.allTerms.count {
//                matrix[row,col] = tfidfVector[row]
//            }
//        }
//        return matrix
//    }
//
//}
