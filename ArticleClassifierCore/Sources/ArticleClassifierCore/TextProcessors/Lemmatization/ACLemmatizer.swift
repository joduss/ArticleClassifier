//
//  File.swift
//  
//
//  Created by Jonathan Duss on 03.01.21.
//

import Foundation

public protocol ACLemmatizer {
    func lemmatize(text: String) -> String
}
