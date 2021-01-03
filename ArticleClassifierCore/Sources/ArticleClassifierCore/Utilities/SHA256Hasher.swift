//
//  File.swift
//  
//
//  Created by Jonathan Duss on 02.05.20.
//

import Foundation
import CryptoKit

public struct SHA256Hasher {
    
    private var hasher = SHA256()
    
    public init() {}
    
    public mutating func append(_ value: String) {
        if let data = value.data(using: .utf8) {
            hasher.update(data: data)
        }
    }
    
    public func finalize() -> String {
        let digest = hasher.finalize()
        
        // Transform each byte into hexadecimal.
        return digest.map({byte in String(format: "%x", byte)}).joined()
    }
}
