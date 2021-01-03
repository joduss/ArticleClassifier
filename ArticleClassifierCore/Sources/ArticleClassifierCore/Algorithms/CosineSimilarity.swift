//
//  CosineSimilarity.swift
//  ArticleClassifierCore-macos
//
//  Created by Jonathan Duss on 03.03.19.
//

import Foundation

public class CosineSimilarity {
    
    
    /// Compute the cosine similarity between 2 vectors
    ///
    /// - Parameters:
    ///   - vector1: first vector
    ///   - vector2: second vector
    /// - Returns: cosine similarity
    public static func compute(vector1: [Double], vector2: [Double]) -> Double {
        
        guard vector1.count == vector2.count else {
            fatalError("Both vector should be of the same size!")
        }
        
        var dotProduct = 0.0
        for idx in 0..<vector1.endIndex {
            dotProduct += vector1[idx] * vector2[idx]
        }
        
        var normSquaredV1 = 0.0
        var normSquaredV2 = 0.0
        for idx in 0..<vector1.endIndex {
            normSquaredV1 += pow(vector1[idx], 2)
            normSquaredV2 += pow(vector2[idx], 2)
        }

        return dotProduct / (sqrt(normSquaredV1) * sqrt(normSquaredV2))
    }
    
    
    public static func compute(vector1: ContiguousArray<Double>, vector2: ContiguousArray<Double>) -> Double {
        
        guard vector1.count == vector2.count else {
            fatalError("Both vector should be of the same size!")
        }
        
        var dotProduct = 0.0
        for idx in 0..<vector1.endIndex {
            dotProduct += vector1[idx] * vector2[idx]
        }
        
        var normSquaredV1 = 0.0
        var normSquaredV2 = 0.0
        for idx in 0..<vector1.endIndex {
            normSquaredV1 += pow(vector1[idx], 2)
            normSquaredV2 += pow(vector2[idx], 2)
        }
        
        return dotProduct / (sqrt(normSquaredV1) * sqrt(normSquaredV2))
    }
}
