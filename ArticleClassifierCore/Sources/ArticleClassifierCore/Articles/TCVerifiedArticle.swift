//
//  File.swift
//  
//
//  Created by Jonathan Duss on 26.01.20.
//

import Foundation


public class TCVerifiedArticle {
    
    public let id: String
    public let title: String
    public let summary: String
    public var themes: [String]
    public var verifiedThemes: [String] = []
    public var predictedThemes: [String] = []

    
    public init(title: String?, summary: String?, themes: [String] = []) {
        self.id = TCVerifiedArticle.hashSha256(title: title ?? "", summary: summary ?? "")
        self.title = title ?? ""
        self.summary = summary ?? ""
        self.themes = themes
    }
    
    public init (dto: TCVerifiedArticleDTO) {
        self.id = dto.id ?? TCVerifiedArticle.hashSha256(title: dto.title, summary: dto.summary)
        self.title = dto.title
        self.summary = dto.summary
        self.themes = dto.themes
        self.verifiedThemes = dto.verifiedThemes
        self.predictedThemes = dto.predictedThemes
    }
    
    private init(id: String, title: String?, summary: String?, themes: [String] = []) {
        self.id = id
        self.title = title ?? ""
        self.summary = summary ?? ""
        self.themes = themes
    }
    
    public func articleWithUpdated(title newTitle: String, summary newSummary: String) -> TCVerifiedArticle {
        let newArticle = TCVerifiedArticle(id: id, title: newTitle, summary: newSummary, themes: themes)
        newArticle.verifiedThemes += verifiedThemes
        newArticle.predictedThemes += predictedThemes
        return newArticle
    }
    
    public static func == (lhs: TCVerifiedArticle, rhs: TCVerifiedArticle) -> Bool {
        return lhs.title == rhs.title && lhs.summary == rhs.title
    }
    
    public func toDto() -> TCVerifiedArticleDTO {
        return TCVerifiedArticleDTO(
            id: id,
            title: title,
            summary: summary,
            themes: themes,
            verifiedThemes: verifiedThemes,
            predictedThemes: predictedThemes)
    }
    
    public func verify(theme: String) {
        guard verifiedThemes.contains(theme) == false else { return }
        
        verifiedThemes.append(theme)
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(title)
        hasher.combine(summary)
    }
    
    public var hashValue: Int {
        var hasher = Hasher()
        hash(into: &hasher)
        return hasher.finalize()
    }
    
    public var titleSummaryHash: Int {
        var hasher = Hasher()
        hasher.combine(title)
        hasher.combine(summary)
        return hasher.finalize()
    }
    
    private static func hashSha256(title: String, summary: String) -> String {
        var hasher = SHA256Hasher()
        hasher.append(title)
        hasher.append(summary)
        return hasher.finalize()
    }
}
