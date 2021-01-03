//
//  ArrayExtension.swift
//  ArticleThemeTagger
//
//  Created by Jonathan Duss on 24.04.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import Foundation

extension Array where Element:Equatable & Hashable {
        
    func containsOne(from array:[Element]) -> Bool {
        for conditionElement in array {
            if contains(conditionElement) {
                return true
            }
        }
                
        return false
    }
    
    func intersection(with array: [Element]) -> Array {
        let setOne = Set(self)
        let setTwo = Set(array)
        
        return Array(setOne.intersection(setTwo))
    }
    
}
