# Block 2: JavaScript Concepts And Coding

## 1. Variables: let, const, var

### const
Cannot be reassigned. Use for values that don't change.

```javascript
const baseUrl = "https://example.com";
const timeout = 30000;

// Error: Cannot reassign
baseUrl = "https://other.com"; // TypeError
```

### let
Can be reassigned. Use for values that change.

```javascript
let retryCount = 0;
retryCount = 1; // OK

let username = "guest";
username = "admin"; // OK
```

### var
Old style. Avoid in modern JavaScript. Function-scoped, not block-scoped.

```javascript
// Avoid var
var oldWay = "don't use";

// Use let or const instead
```

### Block Scope

```javascript
if (true) {
  let x = 10;
  const y = 20;
  var z = 30;
}

console.log(z); // 30 (var is function-scoped, leaks out)
console.log(x); // Error (let is block-scoped)
console.log(y); // Error (const is block-scoped)
```

---

## 2. Data Types

### Primitives

```javascript
// String
const name = "Lokender";
const greeting = `Hello ${name}`; // template literal

// Number
const age = 30;
const price = 99.99;

// Boolean
const isActive = true;
const isAdmin = false;

// Null
const empty = null;

// Undefined
let notAssigned;
console.log(notAssigned); // undefined

// Symbol (rare)
const sym = Symbol("id");
```

### Reference Types

```javascript
// Array
const numbers = [1, 2, 3, 4, 5];
const mixed = [1, "two", true, null];

// Object
const user = {
  name: "Lokender",
  role: "QA Lead",
  age: 30
};

// Function
const greet = function(name) {
  return `Hello ${name}`;
};
```

---

## 3. Functions

### Regular Function

```javascript
function add(a, b) {
  return a + b;
}

console.log(add(2, 3)); // 5
```

### Arrow Function

```javascript
const add = (a, b) => {
  return a + b;
};

// Short form (single expression)
const add = (a, b) => a + b;

// Single parameter (no parentheses needed)
const double = n => n * 2;

// No parameters
const sayHello = () => "Hello";
```

### Default Parameters

```javascript
function greet(name = "Guest") {
  return `Hello ${name}`;
}

greet();        // "Hello Guest"
greet("Lokender"); // "Hello Lokender"
```

### Rest Parameters

```javascript
function sum(...numbers) {
  return numbers.reduce((total, n) => total + n, 0);
}

sum(1, 2, 3);       // 6
sum(1, 2, 3, 4, 5); // 15
```

---

## 4. Arrays

### Creating Arrays

```javascript
const numbers = [1, 2, 3, 4, 5];
const users = [
  { name: "A", role: "admin" },
  { name: "B", role: "viewer" }
];
```

### Accessing Elements

```javascript
const first = numbers[0];     // 1
const last = numbers[numbers.length - 1]; // 5
```

### Array Methods

#### push / pop (end)

```javascript
const arr = [1, 2, 3];
arr.push(4);    // [1, 2, 3, 4]
arr.pop();      // [1, 2, 3], returns 4
```

#### unshift / shift (beginning)

```javascript
const arr = [1, 2, 3];
arr.unshift(0); // [0, 1, 2, 3]
arr.shift();    // [1, 2, 3], returns 0
```

#### map (transform each element)

```javascript
const numbers = [1, 2, 3, 4];
const doubled = numbers.map(n => n * 2);
// [2, 4, 6, 8]

const users = [
  { name: "A", age: 25 },
  { name: "B", age: 30 }
];
const names = users.map(u => u.name);
// ["A", "B"]
```

#### filter (keep elements matching condition)

```javascript
const numbers = [1, 2, 3, 4, 5, 6];
const evens = numbers.filter(n => n % 2 === 0);
// [2, 4, 6]

const users = [
  { name: "A", role: "admin" },
  { name: "B", role: "viewer" },
  { name: "C", role: "admin" }
];
const admins = users.filter(u => u.role === "admin");
// [{ name: "A", role: "admin" }, { name: "C", role: "admin" }]
```

#### find (get first matching element)

```javascript
const users = [
  { id: 1, name: "A" },
  { id: 2, name: "B" },
  { id: 3, name: "C" }
];

const user = users.find(u => u.id === 2);
// { id: 2, name: "B" }

const notFound = users.find(u => u.id === 99);
// undefined
```

#### findIndex

```javascript
const numbers = [10, 20, 30, 40];
const index = numbers.findIndex(n => n === 30);
// 2
```

#### forEach (loop without returning)

```javascript
const numbers = [1, 2, 3];
numbers.forEach(n => console.log(n));
// 1
// 2
// 3
```

#### includes (check if exists)

```javascript
const numbers = [1, 2, 3, 4, 5];
numbers.includes(3);  // true
numbers.includes(10); // false

const roles = ["admin", "viewer", "editor"];
roles.includes("admin"); // true
```

#### some (at least one matches)

```javascript
const numbers = [1, 2, 3, 4, 5];
numbers.some(n => n > 4); // true
numbers.some(n => n > 10); // false
```

#### every (all match)

```javascript
const numbers = [2, 4, 6, 8];
numbers.every(n => n % 2 === 0); // true
numbers.every(n => n > 5);       // false
```

#### reduce (accumulate to single value)

```javascript
const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((total, n) => total + n, 0);
// 15

const max = numbers.reduce((max, n) => n > max ? n : max, numbers[0]);
// 5
```

#### sort

```javascript
// Strings (default)
const names = ["Charlie", "Alice", "Bob"];
names.sort(); // ["Alice", "Bob", "Charlie"]

// Numbers (need compare function)
const numbers = [10, 5, 20, 1];
numbers.sort((a, b) => a - b); // [1, 5, 10, 20] ascending
numbers.sort((a, b) => b - a); // [20, 10, 5, 1] descending

// Objects
const users = [
  { name: "B", age: 30 },
  { name: "A", age: 25 }
];
users.sort((a, b) => a.age - b.age); // by age ascending
```

#### slice (copy portion)

```javascript
const arr = [1, 2, 3, 4, 5];
arr.slice(1, 3);  // [2, 3] (index 1 to 2)
arr.slice(2);     // [3, 4, 5] (from index 2)
arr.slice(-2);    // [4, 5] (last 2)
```

#### splice (remove/add in place)

```javascript
const arr = [1, 2, 3, 4, 5];
arr.splice(2, 1);     // removes 1 element at index 2, arr = [1, 2, 4, 5]
arr.splice(1, 0, 99); // inserts 99 at index 1, arr = [1, 99, 2, 4, 5]
```

#### join

```javascript
const arr = ["a", "b", "c"];
arr.join("");   // "abc"
arr.join("-");  // "a-b-c"
arr.join(", "); // "a, b, c"
```

#### concat

```javascript
const a = [1, 2];
const b = [3, 4];
const c = a.concat(b); // [1, 2, 3, 4]
```

#### Spread operator

```javascript
const a = [1, 2];
const b = [3, 4];
const c = [...a, ...b]; // [1, 2, 3, 4]

const copy = [...a]; // shallow copy
```

---

## 5. Objects

### Creating Objects

```javascript
const user = {
  name: "Lokender",
  role: "QA Lead",
  age: 30,
  isActive: true
};
```

### Accessing Properties

```javascript
// Dot notation
console.log(user.name); // "Lokender"

// Bracket notation
console.log(user["role"]); // "QA Lead"

// Dynamic key
const key = "age";
console.log(user[key]); // 30
```

### Modifying Objects

```javascript
user.name = "Lokender Singh";
user.city = "Noida"; // add new property
delete user.isActive; // remove property
```

### Destructuring

```javascript
const user = { name: "Lokender", role: "QA Lead", age: 30 };

// Extract properties
const { name, role } = user;
console.log(name); // "Lokender"
console.log(role); // "QA Lead"

// Rename
const { name: userName, role: userRole } = user;

// Default value
const { city = "Unknown" } = user;

// Nested
const response = {
  data: {
    user: { name: "A", email: "a@test.com" }
  }
};
const { data: { user: { name: userName } } } = response;
```

### Spread Operator

```javascript
const user = { name: "Lokender", role: "QA" };

// Copy
const copy = { ...user };

// Merge
const updated = { ...user, age: 30 };

// Override
const changed = { ...user, role: "Lead" };
```

### Object Methods

```javascript
const user = { name: "Lokender", role: "QA" };

// Get keys
Object.keys(user);   // ["name", "role"]

// Get values
Object.values(user); // ["Lokender", "QA"]

// Get entries
Object.entries(user); // [["name", "Lokender"], ["role", "QA"]]

// Check if key exists
"name" in user;      // true
user.hasOwnProperty("name"); // true
```

---

## 6. Conditionals

### if / else

```javascript
const role = "admin";

if (role === "admin") {
  console.log("Full access");
} else if (role === "editor") {
  console.log("Edit access");
} else {
  console.log("View only");
}
```

### Ternary Operator

```javascript
const isAdmin = true;
const access = isAdmin ? "Full" : "Limited";

// Nested (avoid if complex)
const role = "admin";
const level = role === "admin" ? 3 : role === "editor" ? 2 : 1;
```

### Logical Operators

```javascript
// AND (both must be true)
if (isLoggedIn && isAdmin) {
  // ...
}

// OR (at least one true)
if (isAdmin || isEditor) {
  // ...
}

// NOT
if (!isGuest) {
  // ...
}

// Nullish coalescing
const name = user.name ?? "Guest"; // uses "Guest" if name is null/undefined

// Optional chaining
const email = user?.profile?.email; // undefined if any part is null/undefined
```

---

## 7. Loops

### for

```javascript
for (let i = 0; i < 5; i++) {
  console.log(i);
}
```

### for...of (arrays, strings)

```javascript
const numbers = [1, 2, 3];
for (const num of numbers) {
  console.log(num);
}

const str = "hello";
for (const char of str) {
  console.log(char);
}
```

### for...in (object keys)

```javascript
const user = { name: "Lokender", role: "QA" };
for (const key in user) {
  console.log(`${key}: ${user[key]}`);
}
```

### while

```javascript
let count = 0;
while (count < 5) {
  console.log(count);
  count++;
}
```

---

## 8. Promises And Async/Await

### What Is A Promise?

A Promise represents a future value from an asynchronous operation.

States:
- **Pending**: initial state
- **Fulfilled**: completed successfully
- **Rejected**: failed

### Creating Promises

```javascript
const promise = new Promise((resolve, reject) => {
  setTimeout(() => {
    const success = true;
    if (success) {
      resolve("Data loaded");
    } else {
      reject("Error loading data");
    }
  }, 1000);
});
```

### Using Promises

```javascript
promise
  .then(data => console.log(data))
  .catch(error => console.error(error))
  .finally(() => console.log("Done"));
```

### Async/Await

```javascript
async function fetchUser() {
  try {
    const response = await fetch("https://api.example.com/user");
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
}

// Usage
const user = await fetchUser();
```

### Multiple Async Operations

```javascript
// Sequential (one after another)
async function sequential() {
  const user = await fetchUser();
  const posts = await fetchPosts(user.id);
  return { user, posts };
}

// Parallel (all at once)
async function parallel() {
  const [users, posts, comments] = await Promise.all([
    fetchUsers(),
    fetchPosts(),
    fetchComments()
  ]);
  return { users, posts, comments };
}
```

---

## 9. Try/Catch

```javascript
try {
  const result = riskyOperation();
  console.log(result);
} catch (error) {
  console.error("Error:", error.message);
} finally {
  console.log("Cleanup");
}
```

### With Async/Await

```javascript
async function fetchData() {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Fetch failed:", error);
    return null;
  }
}
```

---

## 10. Modules (Import/Export)

### Named Export

```javascript
// utils.js
export const add = (a, b) => a + b;
export const subtract = (a, b) => a - b;

// main.js
import { add, subtract } from "./utils.js";
```

### Default Export

```javascript
// LoginPage.js
class LoginPage {
  // ...
}
export default LoginPage;

// test.js
import LoginPage from "./LoginPage.js";
```

### CommonJS (Node.js)

```javascript
// utils.js
const add = (a, b) => a + b;
module.exports = { add };

// main.js
const { add } = require("./utils.js");
```

---

## 11. Coding Problems With Solutions

### Problem 1: Reverse String

```javascript
function reverse(str) {
  return str.split("").reverse().join("");
}

// Alternative
function reverseLoop(str) {
  let reversed = "";
  for (const char of str) {
    reversed = char + reversed;
  }
  return reversed;
}

console.log(reverse("Playwright")); // "thgirwyalP"
```

### Problem 2: Palindrome

```javascript
function isPalindrome(str) {
  const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, "");
  return cleaned === cleaned.split("").reverse().join("");
}

console.log(isPalindrome("A man, a plan, a canal: Panama")); // true
console.log(isPalindrome("hello")); // false
```

### Problem 3: Character Frequency

```javascript
function charFrequency(str) {
  const freq = {};
  for (const char of str.toLowerCase()) {
    if (/[a-z0-9]/.test(char)) {
      freq[char] = (freq[char] || 0) + 1;
    }
  }
  return freq;
}

console.log(charFrequency("Hello!!")); // { h: 1, e: 1, l: 2, o: 1 }
```

### Problem 4: First Non-Repeating Character

```javascript
function firstNonRepeating(str) {
  const freq = {};
  
  for (const char of str) {
    freq[char] = (freq[char] || 0) + 1;
  }
  
  for (const char of str) {
    if (freq[char] === 1) {
      return char;
    }
  }
  
  return null;
}

console.log(firstNonRepeating("swiss")); // "w"
console.log(firstNonRepeating("aabbcc")); // null
```

### Problem 5: First Repeating Character

```javascript
function firstRepeating(str) {
  const seen = new Set();
  
  for (const char of str) {
    if (seen.has(char)) {
      return char;
    }
    seen.add(char);
  }
  
  return null;
}

console.log(firstRepeating("swiss")); // "s"
console.log(firstRepeating("abcdef")); // null
```

### Problem 6: Anagram

```javascript
function isAnagram(a, b) {
  const clean = str => str.toLowerCase().replace(/[^a-z0-9]/g, "");
  const sortedA = clean(a).split("").sort().join("");
  const sortedB = clean(b).split("").sort().join("");
  return sortedA === sortedB;
}

console.log(isAnagram("listen", "silent")); // true
console.log(isAnagram("hello", "world")); // false
```

### Problem 7: Remove Duplicates

```javascript
function removeDuplicates(arr) {
  return [...new Set(arr)];
}

// Preserving order (manual)
function removeDuplicatesManual(arr) {
  const seen = new Set();
  const result = [];
  for (const item of arr) {
    if (!seen.has(item)) {
      result.push(item);
      seen.add(item);
    }
  }
  return result;
}

console.log(removeDuplicates([1, 2, 2, 3, 1, 4])); // [1, 2, 3, 4]
```

### Problem 8: Second Largest

```javascript
function secondLargest(arr) {
  const unique = [...new Set(arr)].sort((a, b) => b - a);
  return unique.length >= 2 ? unique[1] : null;
}

// Single pass
function secondLargestSinglePass(arr) {
  let first = -Infinity;
  let second = -Infinity;
  
  for (const num of arr) {
    if (num > first) {
      second = first;
      first = num;
    } else if (num > second && num !== first) {
      second = num;
    }
  }
  
  return second === -Infinity ? null : second;
}

console.log(secondLargest([10, 5, 20, 20, 8])); // 10
```

### Problem 9: FizzBuzz

```javascript
function fizzBuzz(n) {
  const result = [];
  for (let i = 1; i <= n; i++) {
    if (i % 15 === 0) {
      result.push("FizzBuzz");
    } else if (i % 3 === 0) {
      result.push("Fizz");
    } else if (i % 5 === 0) {
      result.push("Buzz");
    } else {
      result.push(i);
    }
  }
  return result;
}

console.log(fizzBuzz(15));
// [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"]
```

### Problem 10: Factorial

```javascript
function factorial(n) {
  if (n < 0) return null;
  if (n === 0 || n === 1) return 1;
  
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
}

console.log(factorial(5)); // 120
```

### Problem 11: Fibonacci

```javascript
function fibonacci(n) {
  if (n <= 0) return [];
  if (n === 1) return [0];
  
  const result = [0, 1];
  while (result.length < n) {
    result.push(result[result.length - 1] + result[result.length - 2]);
  }
  return result;
}

console.log(fibonacci(8)); // [0, 1, 1, 2, 3, 5, 8, 13]
```

### Problem 12: Common Elements

```javascript
function commonElements(arr1, arr2) {
  const set1 = new Set(arr1);
  return arr2.filter(item => set1.has(item));
}

console.log(commonElements([1, 2, 3, 4], [3, 4, 5, 6])); // [3, 4]
```

### Problem 13: Flatten Array

```javascript
function flatten(arr) {
  return arr.flat(Infinity);
}

// Manual recursive
function flattenManual(arr) {
  const result = [];
  for (const item of arr) {
    if (Array.isArray(item)) {
      result.push(...flattenManual(item));
    } else {
      result.push(item);
    }
  }
  return result;
}

console.log(flatten([1, [2, [3, [4]]]])); // [1, 2, 3, 4]
```

### Problem 14: Check Required Keys

```javascript
function hasRequiredKeys(obj, requiredKeys) {
  return requiredKeys.every(key => key in obj);
}

const user = { id: 1, name: "Lokender", role: "QA" };
console.log(hasRequiredKeys(user, ["id", "name"])); // true
console.log(hasRequiredKeys(user, ["id", "email"])); // false
```

### Problem 15: Filter And Sort Objects

```javascript
const users = [
  { name: "Charlie", age: 35, role: "admin" },
  { name: "Alice", age: 28, role: "viewer" },
  { name: "Bob", age: 32, role: "admin" }
];

// Filter admins
const admins = users.filter(u => u.role === "admin");

// Sort by age
const sortedByAge = [...users].sort((a, b) => a.age - b.age);

// Sort by name
const sortedByName = [...users].sort((a, b) => a.name.localeCompare(b.name));

// Get names of admins sorted by age
const adminNamesSorted = users
  .filter(u => u.role === "admin")
  .sort((a, b) => a.age - b.age)
  .map(u => u.name);

console.log(adminNamesSorted); // ["Bob", "Charlie"]
```

---

## Quick Reference

| Operation | Code |
|-----------|------|
| Check array has item | `arr.includes(item)` |
| Find item in array | `arr.find(x => x.id === 1)` |
| Filter array | `arr.filter(x => x.active)` |
| Transform array | `arr.map(x => x.name)` |
| Check all match | `arr.every(x => x > 0)` |
| Check any match | `arr.some(x => x > 0)` |
| Sum array | `arr.reduce((sum, x) => sum + x, 0)` |
| Sort numbers | `arr.sort((a, b) => a - b)` |
| Remove duplicates | `[...new Set(arr)]` |
| Check key exists | `"key" in obj` |
| Get object keys | `Object.keys(obj)` |
| Get object values | `Object.values(obj)` |
| Copy object | `{ ...obj }` |
| Merge objects | `{ ...obj1, ...obj2 }` |
