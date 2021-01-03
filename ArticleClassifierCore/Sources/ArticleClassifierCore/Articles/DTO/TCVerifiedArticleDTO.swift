//
//  File.swift
//  
//
//  Created by Jonathan Duss on 02.05.20.
//

import Foundation

public struct TCVerifiedArticleDTO: Codable, Hashable {
    
    public var id: String?
    public let title: String
    public let summary: String
    public var themes: [String]
    public var verifiedThemes: [String] = []
    public var predictedThemes: [String] = []
    
    private enum CodingKeys: String, CodingKey {
        case title
        case summary
        case themes
        case verifiedThemes
        case predictedThemes
        case id
    }
}
