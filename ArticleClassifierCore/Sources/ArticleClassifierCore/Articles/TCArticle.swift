//
//  ArticleForIO.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 28.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation

public struct TCArticle: Codable, Hashable {

    public let title: String
    public let summary: String
    public let themes: [String]
    public let associatedObject: AnyHashable?
    
    private enum CodingKeys: String, CodingKey {
        case title
        case summary
        case themes
    }
    
    public init(title: String?, summary: String?, themes: [String] = [], associatedObject: AnyHashable? = nil) {
        self.title = title ?? ""
        self.summary = summary ?? ""
        self.themes = themes
        self.associatedObject = associatedObject
    }
    
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        title = try values.decode(String.self, forKey: .title)
        summary = try values.decode(String.self, forKey: .summary)
        themes = try values.decode([String].self, forKey: .themes)
        associatedObject = nil
    }
    
    
    public static func == (lhs: TCArticle, rhs: TCArticle) -> Bool {
        return (lhs.title + lhs.summary).hashValue == (rhs.title + rhs.summary).hashValue
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(title + summary)
    }
}
